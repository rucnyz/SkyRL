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
import os
from pathlib import Path
from typing import Any, ClassVar

import httpx
from e2b import AsyncTemplate, Template
from e2b.exceptions import SandboxException

from harbor.environments.e2b import E2BEnvironment
from harbor.models.trial.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

_E2B_API_BASE = "https://api.e2b.app"

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

    async def _create_sandbox(self) -> None:
        """Wrap upstream sandbox.create so a mid-flight HTTP failure (the
        SDK never returns a sandbox object even though e2b already started
        provisioning one) doesn't leak a sandbox on the account quota.
        """
        try:
            await super()._create_sandbox()
        except Exception:
            # Re-raise after firing the reaper so the trial's retry logic
            # still gets to see the original exception.
            await self._reap_zombie_by_env_name()
            raise

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
