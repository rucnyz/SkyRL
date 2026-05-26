"""Mock-based verification of the owner-tracked sandbox reaper in
``SharedTemplateE2BEnvironment``.

No GPU / no e2b backend required. Stubs out:
  - ``e2b.AsyncSandbox.create`` (fake sandbox)
  - ``httpx.AsyncClient`` GET /sandboxes + DELETE /sandboxes/{id}
  - The parent ``harbor.environments.e2b.E2BEnvironment.stop`` (so we don't
    pull in the full harbor trial lifecycle)

Covers the scenarios that matter for correctness:
  1. Normal create→stop: registry updated, backup DELETE called.
  2. SDK kill silently fails: registry still deregistered, backup DELETE
     issued (no zombie leak).
  3. Owner reaper deletes orphans (env_name not in registry, our PID) and
     preserves live entries (env_name in registry) and foreign entries
     (different PID, no PID, etc.).
  4. mid-flight create failure: zombie reaper-by-env-name path triggered.

Run:
    uv run python -m examples.train_integrations.harbor_pgc.tests.test_owner_reaper
"""

from __future__ import annotations

import asyncio
import os
import sys
import types
import unittest
from unittest import mock


# --------------------------------------------------------------------------
# Module-import shim. ``skyrl_e2b`` imports a lot of harbor + e2b internals
# we don't have on the test path; replace them with minimal stubs.
# --------------------------------------------------------------------------


def _install_stub_modules() -> None:
    """Install stub modules so importing ``skyrl_e2b`` works without the
    real harbor / e2b packages."""
    # Stub e2b module tree
    e2b_mod = types.ModuleType("e2b")
    e2b_mod.AsyncSandbox = mock.MagicMock()
    e2b_mod.AsyncTemplate = mock.MagicMock()
    e2b_mod.Template = mock.MagicMock()
    sys.modules["e2b"] = e2b_mod

    exc_mod = types.ModuleType("e2b.exceptions")
    class SandboxException(Exception):
        pass
    exc_mod.SandboxException = SandboxException
    sys.modules["e2b.exceptions"] = exc_mod

    # Stub tenacity (only the decorators we use; pass-through)
    tenacity_mod = types.ModuleType("tenacity")
    def retry(*_a, **_kw):
        def deco(fn):
            return fn
        return deco
    def stop_after_attempt(_n):
        return None
    def wait_exponential(*_a, **_kw):
        return None
    tenacity_mod.retry = retry
    tenacity_mod.stop_after_attempt = stop_after_attempt
    tenacity_mod.wait_exponential = wait_exponential
    sys.modules["tenacity"] = tenacity_mod

    # Stub harbor.environments.e2b
    harbor_mod = types.ModuleType("harbor")
    harbor_envs_mod = types.ModuleType("harbor.environments")
    harbor_e2b_mod = types.ModuleType("harbor.environments.e2b")

    class FakeBaseE2BEnvironment:
        """Minimal stand-in for harbor's E2BEnvironment so we can call
        super() in the subclass without pulling harbor."""
        def __init__(self, environment_dir=None, environment_name="env-test",
                     session_id="sess-test", trial_paths=None,
                     task_env_config=None, **_kwargs):
            self.environment_name = environment_name
            self.session_id = session_id
            self.task_env_config = task_env_config
            self.trial_paths = trial_paths
            self._sandbox = None
            self.logger = mock.MagicMock()

        async def _create_sandbox(self):
            raise NotImplementedError

        async def stop(self, delete: bool) -> None:
            # Mirror harbor's behavior: swallow exception, null out reference.
            if self._sandbox is not None:
                try:
                    if hasattr(self._sandbox, "kill"):
                        await self._sandbox.kill()
                except Exception:
                    pass
                self._sandbox = None

    harbor_e2b_mod.E2BEnvironment = FakeBaseE2BEnvironment
    sys.modules["harbor"] = harbor_mod
    sys.modules["harbor.environments"] = harbor_envs_mod
    sys.modules["harbor.environments.e2b"] = harbor_e2b_mod

    # Stub harbor.models.trial.config / paths
    config_mod = types.ModuleType("harbor.models.trial.config")
    class EnvironmentConfig:
        docker_image = "ghcr.io/test/img:1"
        cpus = 1
        memory_mb = 1024
        allow_internet = True
    config_mod.EnvironmentConfig = EnvironmentConfig
    sys.modules["harbor.models"] = types.ModuleType("harbor.models")
    sys.modules["harbor.models.trial"] = types.ModuleType("harbor.models.trial")
    sys.modules["harbor.models.trial.config"] = config_mod
    paths_mod = types.ModuleType("harbor.models.trial.paths")
    class TrialPaths:
        pass
    paths_mod.TrialPaths = paths_mod.TrialPaths = TrialPaths
    sys.modules["harbor.models.trial.paths"] = paths_mod


_install_stub_modules()

# Now safe to import the unit under test.
sys.path.insert(0, "/scratch/yuzhou/projects/SkyRL")
from examples.train_integrations.harbor_pgc.environments import skyrl_e2b  # noqa: E402


# --------------------------------------------------------------------------
# Test helpers
# --------------------------------------------------------------------------


class FakeSandbox:
    def __init__(self, sid: str, kill_should_fail: bool = False):
        self.sandbox_id = sid
        self._kill_should_fail = kill_should_fail
        self.kill_calls = 0

    async def kill(self):
        self.kill_calls += 1
        if self._kill_should_fail:
            raise RuntimeError("simulated SDK kill failure (5xx)")


class FakeAsyncClient:
    """Replaces httpx.AsyncClient. Records every GET/DELETE and serves
    a configurable list of sandboxes for GET /sandboxes."""

    def __init__(self, *_a, **_kw):
        pass

    sandboxes: list[dict] = []
    delete_calls: list[str] = []
    delete_response_status: int = 204
    get_response_status: int = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return None

    async def get(self, url, headers=None):
        resp = mock.MagicMock()
        resp.status_code = FakeAsyncClient.get_response_status
        resp.json = lambda: FakeAsyncClient.sandboxes
        return resp

    async def delete(self, url, headers=None):
        sid = url.rsplit("/", 1)[-1]
        FakeAsyncClient.delete_calls.append(sid)
        resp = mock.MagicMock()
        resp.status_code = FakeAsyncClient.delete_response_status
        return resp

    @classmethod
    def reset(cls):
        cls.sandboxes = []
        cls.delete_calls = []
        cls.delete_response_status = 204
        cls.get_response_status = 200


def _new_env(name: str, docker_image: str = "ghcr.io/test/img:1") -> "skyrl_e2b.SharedTemplateE2BEnvironment":
    from harbor.models.trial.config import EnvironmentConfig
    cfg = EnvironmentConfig()
    cfg.docker_image = docker_image
    env = skyrl_e2b.SharedTemplateE2BEnvironment(
        environment_dir=None,
        environment_name=name,
        session_id=f"sess-{name}",
        trial_paths=None,
        task_env_config=cfg,
    )
    return env


def _reset_globals():
    """Reset module-level state between tests."""
    skyrl_e2b._LIVE_ENVIRONMENT_NAMES.clear()
    if skyrl_e2b._OWNER_REAPER_TASK is not None:
        skyrl_e2b._OWNER_REAPER_TASK.cancel()
    skyrl_e2b._OWNER_REAPER_TASK = None
    skyrl_e2b._LIVE_ENVIRONMENT_LOCK = None
    FakeAsyncClient.reset()


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


class TestOwnerReaper(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        _reset_globals()
        os.environ["E2B_API_KEY"] = "test-key"

    async def asyncTearDown(self):
        if skyrl_e2b._OWNER_REAPER_TASK is not None:
            skyrl_e2b._OWNER_REAPER_TASK.cancel()
            try:
                await skyrl_e2b._OWNER_REAPER_TASK
            except (asyncio.CancelledError, Exception):
                pass
        skyrl_e2b._OWNER_REAPER_TASK = None

    # ---------------- create / stop happy path ----------------

    async def test_create_registers_and_stamps_owner_pid(self):
        fake_sbx = FakeSandbox("sbx-A")
        captured_metadata = {}

        async def fake_create(**kwargs):
            captured_metadata.update(kwargs.get("metadata", {}))
            return fake_sbx

        with mock.patch.object(skyrl_e2b.AsyncSandbox, "create", side_effect=fake_create):
            env = _new_env("env-A")
            await env._create_sandbox()

        self.assertEqual(captured_metadata.get("environment_name"), "env-A")
        self.assertEqual(captured_metadata.get("session_id"), "sess-env-A")
        self.assertEqual(captured_metadata.get("owner_pid"), skyrl_e2b._OWNER_PID)
        self.assertIn("env-A", skyrl_e2b._LIVE_ENVIRONMENT_NAMES)

    async def test_stop_deregisters_and_backup_deletes(self):
        fake_sbx = FakeSandbox("sbx-A")
        async def _async_returns_fake(**_kw):
            return fake_sbx
        with mock.patch.object(skyrl_e2b.AsyncSandbox, "create", side_effect=_async_returns_fake), \
             mock.patch.object(skyrl_e2b.httpx, "AsyncClient", FakeAsyncClient):
            env = _new_env("env-A")
            await env._create_sandbox()
            self.assertIn("env-A", skyrl_e2b._LIVE_ENVIRONMENT_NAMES)

            await env.stop(delete=True)

            self.assertNotIn("env-A", skyrl_e2b._LIVE_ENVIRONMENT_NAMES)
            # SDK kill ran:
            self.assertEqual(fake_sbx.kill_calls, 1)
            # Backup REST DELETE ALSO ran (defense in depth):
            self.assertIn("sbx-A", FakeAsyncClient.delete_calls)

    async def test_stop_swallows_sdk_kill_failure_and_backup_delete_still_runs(self):
        fake_sbx = FakeSandbox("sbx-B", kill_should_fail=True)
        async def _async_returns_fake(**_kw):
            return fake_sbx
        with mock.patch.object(skyrl_e2b.AsyncSandbox, "create", side_effect=_async_returns_fake), \
             mock.patch.object(skyrl_e2b.httpx, "AsyncClient", FakeAsyncClient):
            env = _new_env("env-B")
            await env._create_sandbox()
            await env.stop(delete=True)

            self.assertNotIn("env-B", skyrl_e2b._LIVE_ENVIRONMENT_NAMES)
            # SDK kill was attempted (and failed silently):
            self.assertEqual(fake_sbx.kill_calls, 1)
            # Backup REST DELETE ran and would have caught the zombie:
            self.assertIn("sbx-B", FakeAsyncClient.delete_calls)

    # ---------------- create failure path ----------------

    async def test_create_failure_fires_zombie_reaper_and_no_registry(self):
        async def fake_create(**_kw):
            raise RuntimeError("simulated create 5xx")

        with mock.patch.object(skyrl_e2b.AsyncSandbox, "create", side_effect=fake_create), \
             mock.patch.object(skyrl_e2b.httpx, "AsyncClient", FakeAsyncClient):
            FakeAsyncClient.sandboxes = [
                {"sandboxID": "leak-1", "metadata": {"environment_name": "env-C"}},
            ]
            env = _new_env("env-C")
            with self.assertRaises(RuntimeError):
                await env._create_sandbox()

        self.assertNotIn("env-C", skyrl_e2b._LIVE_ENVIRONMENT_NAMES)
        # The zombie reaper-by-env-name MUST have DELETE'd the leak:
        self.assertIn("leak-1", FakeAsyncClient.delete_calls)

    # ---------------- owner reaper correctness ----------------

    async def test_owner_reaper_kills_only_orphans_we_own(self):
        # Setup: pretend our process knows env-LIVE; env-DEAD was registered
        # then deregistered (Trial finished).
        async with skyrl_e2b._get_live_lock() if skyrl_e2b._LIVE_ENVIRONMENT_LOCK else asyncio.Lock():
            pass  # noop to init the lock
        async with skyrl_e2b._get_live_lock():
            skyrl_e2b._LIVE_ENVIRONMENT_NAMES.add("env-LIVE")

        FakeAsyncClient.sandboxes = [
            {  # OURS + alive trial → KEEP
                "sandboxID": "sbx-live",
                "metadata": {
                    "environment_name": "env-LIVE",
                    "owner_pid": skyrl_e2b._OWNER_PID,
                },
            },
            {  # OURS + dead trial → KILL
                "sandboxID": "sbx-orphan",
                "metadata": {
                    "environment_name": "env-DEAD",
                    "owner_pid": skyrl_e2b._OWNER_PID,
                },
            },
            {  # Different PID → KEEP (another process's sandbox)
                "sandboxID": "sbx-other-proc",
                "metadata": {
                    "environment_name": "env-OTHER",
                    "owner_pid": "9999999",
                },
            },
            {  # No PID stamp at all → KEEP (pre-fix sandbox or foreign tool)
                "sandboxID": "sbx-untagged",
                "metadata": {"environment_name": "env-UNTAGGED"},
            },
            {  # OURS but no environment_name → KEEP (defensive)
                "sandboxID": "sbx-no-env-name",
                "metadata": {"owner_pid": skyrl_e2b._OWNER_PID},
            },
        ]

        with mock.patch.object(skyrl_e2b.httpx, "AsyncClient", FakeAsyncClient):
            # Run one iteration of the reaper directly (not via background
            # task — easier to assert on)
            task = asyncio.create_task(skyrl_e2b._owner_reaper_loop(interval_sec=0.01))
            # Let it run a couple ticks
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self.assertIn("sbx-orphan", FakeAsyncClient.delete_calls)
        self.assertNotIn("sbx-live", FakeAsyncClient.delete_calls)
        self.assertNotIn("sbx-other-proc", FakeAsyncClient.delete_calls)
        self.assertNotIn("sbx-untagged", FakeAsyncClient.delete_calls)
        self.assertNotIn("sbx-no-env-name", FakeAsyncClient.delete_calls)

    async def test_owner_reaper_started_idempotently(self):
        """Multiple _create_sandbox calls must not spawn multiple reaper tasks."""
        async def _create(**_kw):
            return FakeSandbox("x")
        with mock.patch.object(skyrl_e2b.AsyncSandbox, "create", side_effect=_create):
            env1 = _new_env("env-X1")
            await env1._create_sandbox()
            task_after_first = skyrl_e2b._OWNER_REAPER_TASK

            env2 = _new_env("env-X2")
            await env2._create_sandbox()
            task_after_second = skyrl_e2b._OWNER_REAPER_TASK

        self.assertIsNotNone(task_after_first)
        self.assertIs(task_after_first, task_after_second)

    # ---------------- end-to-end: create succeeds, stop fails, reaper cleans ----------------

    async def test_e2e_stop_fails_owner_reaper_catches_orphan(self):
        """The exact production failure mode that motivated this whole thing."""
        fake_sbx = FakeSandbox("sbx-prod", kill_should_fail=True)

        async def fake_create(**_kw):
            return fake_sbx

        # Simulate the env name's sandbox still being live on e2b after
        # stop() silently failed to kill it.
        FakeAsyncClient.sandboxes = [
            {
                "sandboxID": "sbx-prod",
                "metadata": {
                    "environment_name": "env-PROD",
                    "owner_pid": skyrl_e2b._OWNER_PID,
                },
            },
        ]

        with mock.patch.object(skyrl_e2b.AsyncSandbox, "create", side_effect=fake_create), \
             mock.patch.object(skyrl_e2b.httpx, "AsyncClient", FakeAsyncClient):
            env = _new_env("env-PROD")
            await env._create_sandbox()
            await env.stop(delete=True)  # SDK kill fails silently; backup
            # REST delete ALSO returns success here — so we'd already kill it.
            # Clear the call log to test the reaper independently
            FakeAsyncClient.delete_calls.clear()

            # Now simulate that even the backup DELETE failed — sandbox
            # still alive. Reaper must catch it.
            task = asyncio.create_task(skyrl_e2b._owner_reaper_loop(interval_sec=0.01))
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # env-PROD was deregistered by stop(); reaper sees orphan and kills.
        self.assertIn("sbx-prod", FakeAsyncClient.delete_calls)


if __name__ == "__main__":
    unittest.main(verbosity=2)
