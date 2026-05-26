"""E2B environment subclass that shares one template alias per docker_image.

Why we need this
----------------
Nemotron-Terminal-Synthetic-Tasks ships ~1000 tasks per skill, each with:
  - an identical ``environment/Dockerfile`` (per skill) ending in
    ``COPY files/ /app/`` to bake task-specific input data into the image
  - a unique ``environment/files/`` directory

Harbor's stock ``E2BEnvironment._template_name`` hashes the entire
``environment/`` dir (Dockerfile + files/), so every task would get its
own e2b template alias — 5984 template builds per dataset, each minutes
long, just to re-bake the same skill base image with different files/.
That's hours of wasted upfront work for every run.

We instead pre-build one skill-base image per Dockerfile (no
``COPY files/``) and push to a public registry (e.g.
``ghcr.io/rucnyz/nemotron-<skill>:1.0``). At trial time the
sandbox boots from that image, then we upload the task's
``environment/files/`` into ``/app/`` over the e2b SDK — recovering the
``COPY files/ /app/`` semantics that the original Dockerfile would have
produced.

Wiring
------
Loaded via harbor's official ``environment.import_path`` extension
point:

    environment:
      type: e2b
      import_path: "examples.train_integrations.harbor_pgc.environments.skyrl_e2b:SharedTemplateE2BEnvironment"

``HarborGenerator`` injects this when it builds the trial config.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from pathlib import Path
from typing import Any, ClassVar

import httpx
from e2b import AsyncSandbox, AsyncTemplate, Template
from e2b.exceptions import SandboxException
from tenacity import retry, stop_after_attempt, wait_exponential

from harbor.environments.e2b import E2BEnvironment
from harbor.models.trial.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

_E2B_API_BASE = "https://api.e2b.app"

# ---------------------------------------------------------------------------
# Owner-tracked sandbox reaping
# ---------------------------------------------------------------------------
# Why this exists
# ---------------
# Harbor's ``E2BEnvironment.stop()`` calls ``self._sandbox.kill()`` through the
# e2b SDK. If that SDK call hits an HTTP 5xx / transient network error and
# exhausts tenacity's 2-attempt retry budget, ``stop()`` logs an ERROR but the
# ``finally`` block still drops the local reference (``self._sandbox = None``).
# The sandbox on the e2b side never received its DELETE — it lingers until
# e2b's own inactivity_timeout reaps it (observed: hours-to-days). On a 100-
# sandbox account cap, a steady drip of these zombies starves throughput.
#
# This module adds a process-wide registry + background reaper that
# deterministically detects orphans without relying on idle-time heuristics:
#
#   - Each successful sandbox create registers the trial's ``environment_name``
#     in ``_LIVE_ENVIRONMENT_NAMES``.
#   - The metadata stamped onto the sandbox includes ``owner_pid`` so the
#     reaper only ever touches sandboxes our process created (multiple
#     concurrent runs sharing an E2B account are safe).
#   - A background task lists e2b sandboxes every ``_REAPER_INTERVAL_SEC``;
#     anything tagged with our PID whose ``environment_name`` is NOT in the
#     live registry is guaranteed orphaned (the Trial that created it called
#     ``stop()`` already, or its Python object was GC'd) and gets a REST
#     DELETE.
#   - ``stop()`` also fires a backup REST DELETE in case the SDK kill silently
#     swallowed an exception.

logger = logging.getLogger(__name__)

_LIVE_ENVIRONMENT_NAMES: set[str] = set()
_LIVE_ENVIRONMENT_LOCK: asyncio.Lock | None = None  # lazily init in current event loop
_OWNER_REAPER_TASK: asyncio.Task[None] | None = None
_REAPER_INTERVAL_SEC = 60.0
_OWNER_PID = str(os.getpid())


def _get_live_lock() -> asyncio.Lock:
    global _LIVE_ENVIRONMENT_LOCK
    if _LIVE_ENVIRONMENT_LOCK is None:
        _LIVE_ENVIRONMENT_LOCK = asyncio.Lock()
    return _LIVE_ENVIRONMENT_LOCK


async def _owner_reaper_loop(interval_sec: float = _REAPER_INTERVAL_SEC) -> None:
    """Periodically delete e2b sandboxes that this process created but whose
    owning Trial has finished. Runs until cancelled."""
    api_key = os.environ.get("E2B_API_KEY")
    if not api_key:
        return
    while True:
        try:
            await asyncio.sleep(interval_sec)
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"{_E2B_API_BASE}/sandboxes",
                    headers={"X-API-Key": api_key},
                )
                if resp.status_code != 200:
                    continue
                async with _get_live_lock():
                    live = set(_LIVE_ENVIRONMENT_NAMES)
                killed = 0
                for sbx in resp.json():
                    meta = sbx.get("metadata") or {}
                    # Only touch sandboxes this process created. Other
                    # PGC processes on the same E2B account get their own
                    # reaper.
                    if meta.get("owner_pid") != _OWNER_PID:
                        continue
                    env_name = meta.get("environment_name")
                    if not env_name:
                        # We always stamp environment_name; absent means
                        # foreign or pre-this-fix. Leave alone.
                        continue
                    if env_name in live:
                        # Trial that created it is still active.
                        continue
                    sid = sbx.get("sandboxID")
                    if not sid:
                        continue
                    try:
                        del_resp = await client.delete(
                            f"{_E2B_API_BASE}/sandboxes/{sid}",
                            headers={"X-API-Key": api_key},
                        )
                        # 204 = killed, 404 = already gone (race with SDK
                        # finally completing). Both fine.
                        if del_resp.status_code in (204, 404):
                            killed += 1
                    except Exception:  # noqa: BLE001 — best-effort
                        pass
                if killed:
                    logger.warning(
                        f"Owner reaper killed {killed} orphan e2b sandbox(es) "
                        f"(owner_pid={_OWNER_PID}, "
                        f"live_envs={len(live)})."
                    )
        except asyncio.CancelledError:
            return
        except Exception as exc:  # noqa: BLE001 — never crash the loop
            logger.debug(f"Owner reaper iteration failed: {exc}")


def _ensure_owner_reaper_started() -> None:
    """Idempotently start the background owner reaper in the current event
    loop. Called from every successful _create_sandbox so the first
    successful sandbox kicks it off; subsequent calls are no-ops."""
    global _OWNER_REAPER_TASK
    if _OWNER_REAPER_TASK is not None and not _OWNER_REAPER_TASK.done():
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    _OWNER_REAPER_TASK = loop.create_task(_owner_reaper_loop())
    _OWNER_REAPER_TASK.add_done_callback(_clear_reaper_task_on_done)


def _clear_reaper_task_on_done(task: asyncio.Task[None]) -> None:
    global _OWNER_REAPER_TASK
    if _OWNER_REAPER_TASK is task:
        _OWNER_REAPER_TASK = None

# Backoff schedule (seconds) for retrying super().start() when the freshly-built
# template's ``default`` tag is registered server-side but the underlying image
# push is still propagating. Observed propagation gap with first-time builds of
# the 11 Nemotron skill base images is ~10-30s; the long tail covers worst-case
# e2b backend latency without blocking the trial indefinitely.
_TAG_READY_BACKOFFS = (2, 4, 8, 16, 30, 60, 60)


class SharedTemplateE2BEnvironment(E2BEnvironment):
    """E2BEnvironment that shares one template alias across all tasks
    pointing at the same docker_image.
    """

    # Where in the sandbox the task expects its per-trial files to land.
    # Matches the WORKDIR + ``COPY files/ /app/`` contract baked into every
    # original Nemotron Dockerfile we strip out at image-build time.
    _PER_TRIAL_FILES_TARGET = "/app"

    # Serialize template build across concurrent trials that share the same
    # alias. Without this, N trials all see alias_exists=False, all POST a
    # build, only one wins alias registration (rest 403), and trials racing
    # past on the winning alias hit "tag 'default' does not exist" because
    # the actual image push hasn't finished. The lock collapses N concurrent
    # builds into 1 build + (N-1) cache hits.
    _TEMPLATE_BUILD_LOCKS: ClassVar[dict[str, asyncio.Lock]] = {}
    _TEMPLATE_BUILD_LOCKS_MUTEX: ClassVar[asyncio.Lock] = asyncio.Lock()

    @classmethod
    async def _get_template_lock(cls, name: str) -> asyncio.Lock:
        async with cls._TEMPLATE_BUILD_LOCKS_MUTEX:
            lock = cls._TEMPLATE_BUILD_LOCKS.get(name)
            if lock is None:
                lock = asyncio.Lock()
                cls._TEMPLATE_BUILD_LOCKS[name] = lock
            return lock

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            *args,
            **kwargs,
        )

        docker_image = task_env_config.docker_image
        if not docker_image:
            raise ValueError(
                "SharedTemplateE2BEnvironment requires task.toml "
                "[environment].docker_image to point at a pre-built image "
                "(e.g. ghcr.io/rucnyz/nemotron-<skill>:1.0). Run "
                "scripts/rewrite_task_dockerimage.py first."
            )
        # One alias per (sanitised) docker_image URI. Hash the URI so the alias
        # stays under e2b's name length cap and is filesystem-safe.
        image_hash = hashlib.sha256(docker_image.encode()).hexdigest()[:12]
        self._template_name = f"pgc-shared-{image_hash}"

    async def _reap_zombie_by_env_name(self) -> int:
        """List E2B sandboxes and DELETE any whose metadata.environment_name
        matches this trial's. Catches the case where ``POST /sandboxes``
        returned a 5xx / connection drop AFTER the sandbox was created
        server-side — in that path the SDK never produces a sandbox object, so
        ``trial.stop()`` can't kill it, and it sits on the 100-account quota
        for its ``inactivity_timeout``. Best-effort: never raises.
        """
        api_key = os.environ.get("E2B_API_KEY")
        if not api_key:
            return 0
        killed = 0
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{_E2B_API_BASE}/sandboxes",
                    headers={"X-API-Key": api_key},
                )
                if resp.status_code != 200:
                    return 0
                for sbx in resp.json():
                    meta = sbx.get("metadata") or {}
                    if meta.get("environment_name") != self.environment_name:
                        continue
                    sid = sbx.get("sandboxID")
                    if not sid:
                        continue
                    try:
                        await client.delete(
                            f"{_E2B_API_BASE}/sandboxes/{sid}",
                            headers={"X-API-Key": api_key},
                        )
                        killed += 1
                    except Exception:  # noqa: BLE001 — reaper is best-effort
                        pass
        except Exception as exc:  # noqa: BLE001
            self.logger.debug(f"Zombie reaper for {self.environment_name} failed: {exc}")
        if killed:
            self.logger.warning(
                f"Reaped {killed} zombie e2b sandbox(es) for env "
                f"{self.environment_name} (sandbox.create dropped mid-flight)."
            )
        return killed

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _create_sandbox(self) -> None:
        """Override harbor's default to:
          - Stamp ``owner_pid`` into sandbox metadata so the periodic owner
            reaper can distinguish sandboxes this process created from those
            created by other PGC processes sharing the same E2B account.
          - Register the trial's ``environment_name`` in the process-wide
            live set immediately after a successful create.
          - Fire the existing zombie reaper on a mid-flight failure (a
            5xx / dropped connection that the SDK couldn't retry through).
        """
        metadata = {
            "environment_name": self.environment_name,
            "session_id": self.session_id,
            "owner_pid": _OWNER_PID,
        }
        try:
            self._sandbox = await AsyncSandbox.create(
                template=self._template_name,
                metadata=metadata,
                timeout=86_400,
                allow_internet_access=self.task_env_config.allow_internet,
            )
            async with _get_live_lock():
                _LIVE_ENVIRONMENT_NAMES.add(self.environment_name)
            _ensure_owner_reaper_started()
        except Exception:
            # Re-raise after firing the reaper so the trial's retry logic
            # still gets to see the original exception.
            await self._reap_zombie_by_env_name()
            raise

    async def stop(self, delete: bool) -> None:
        """Augments harbor's ``stop()`` with two safety nets:
          1. Always deregister our ``environment_name`` from the live set so
             the owner reaper knows this trial is done (regardless of whether
             the SDK kill succeeded).
          2. Backup REST DELETE in case ``super().stop()`` caught an SDK
             error and silently moved on, leaving the sandbox alive on e2b.
        """
        sandbox_id: str | None = None
        if self._sandbox is not None:
            sandbox_id = (
                getattr(self._sandbox, "sandbox_id", None)
                or getattr(self._sandbox, "sandboxID", None)
            )
        try:
            await super().stop(delete)
        finally:
            async with _get_live_lock():
                _LIVE_ENVIRONMENT_NAMES.discard(self.environment_name)
            if sandbox_id:
                await self._force_delete_via_rest(sandbox_id)

    async def _force_delete_via_rest(self, sandbox_id: str) -> None:
        """Direct REST DELETE that bypasses the SDK's retry/state. Idempotent
        — a 404 (sandbox already gone) is treated as success. Best-effort:
        never raises."""
        api_key = os.environ.get("E2B_API_KEY")
        if not api_key:
            return
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.delete(
                    f"{_E2B_API_BASE}/sandboxes/{sandbox_id}",
                    headers={"X-API-Key": api_key},
                )
                if resp.status_code not in (204, 404):
                    logger.debug(
                        f"Backup REST DELETE {sandbox_id} returned "
                        f"{resp.status_code}; owner reaper will retry."
                    )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.debug(f"Backup REST DELETE {sandbox_id} failed: {exc}")

    async def _create_template(self) -> None:
        """Always build from the public registry image, ignoring any
        ``environment/Dockerfile`` left on disk (NVIDIA's stock Dockerfile
        ends in ``COPY files/ /app/`` which would re-bake per-task data,
        defeating the shared-template optimisation).
        """
        template = Template().from_image(image=self.task_env_config.docker_image)
        await AsyncTemplate.build(
            template=template,
            alias=self._template_name,
            cpu_count=self.task_env_config.cpus,
            memory_mb=self.task_env_config.memory_mb,
        )

    async def start(self, force_build: bool) -> None:
        """Bring the sandbox up via the shared template, then upload the
        per-task ``environment/files/`` directory into ``/app/`` — the
        runtime replacement for the ``COPY files/ /app/`` line we stripped
        from the pre-built image.
        """
        # Per-template lock collapses concurrent builds into one.
        lock = await self._get_template_lock(self._template_name)
        async with lock:
            if force_build or not await self._does_template_exist():
                self.logger.info(f"Building shared template {self._template_name}")
                await self._create_template()

        # Template exists. But: ``AsyncTemplate.build`` returns once the build
        # status flips to "ready", which can precede the moment the image's
        # ``default`` tag is actually pushable. Retry super().start() on the
        # specific 404 until propagation catches up. Other SandboxException
        # subtypes (RateLimitException, etc.) propagate immediately.
        for attempt, backoff in enumerate((*_TAG_READY_BACKOFFS, None)):
            try:
                await super().start(force_build=False)
                break
            except SandboxException as exc:
                if "tag 'default' does not exist" not in str(exc):
                    raise
                if backoff is None:
                    raise RuntimeError(
                        f"Template {self._template_name} never became "
                        f"sandbox-ready after {attempt} retries"
                    ) from exc
                self.logger.warning(
                    f"Template {self._template_name} tag not yet pushable "
                    f"(attempt {attempt + 1}); sleeping {backoff}s"
                )
                await asyncio.sleep(backoff)

        files_dir = self.environment_dir / "files"
        if files_dir.is_dir() and any(files_dir.iterdir()):
            # Mirror the COPY semantics: contents of files/ land directly
            # under /app/ (not nested in /app/files/).
            await self.upload_dir(
                source_dir=files_dir,
                target_dir=self._PER_TRIAL_FILES_TARGET,
            )
