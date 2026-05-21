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
``ghcr.io/rucnyz/pgc-nemotron-<skill>:1.0``). At trial time the
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

import hashlib
from pathlib import Path
from typing import Any

from e2b import AsyncTemplate, Template

from harbor.environments.e2b import E2BEnvironment
from harbor.models.trial.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths


class SharedTemplateE2BEnvironment(E2BEnvironment):
    """E2BEnvironment that shares one template alias across all tasks
    pointing at the same docker_image.
    """

    # Where in the sandbox the task expects its per-trial files to land.
    # Matches the WORKDIR + ``COPY files/ /app/`` contract baked into every
    # original Nemotron Dockerfile we strip out at image-build time.
    _PER_TRIAL_FILES_TARGET = "/app"

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
                "(e.g. ghcr.io/rucnyz/pgc-nemotron-<skill>:1.0). Run "
                "scripts/rewrite_task_dockerimage.py first."
            )
        # One alias per (sanitised) docker_image URI. Hash the URI so the alias
        # stays under e2b's name length cap and is filesystem-safe.
        image_hash = hashlib.sha256(docker_image.encode()).hexdigest()[:12]
        self._template_name = f"pgc-shared-{image_hash}"

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
        await super().start(force_build=force_build)

        files_dir = self.environment_dir / "files"
        if files_dir.is_dir() and any(files_dir.iterdir()):
            # Mirror the COPY semantics: contents of files/ land directly
            # under /app/ (not nested in /app/files/).
            await self.upload_dir(
                source_dir=files_dir,
                target_dir=self._PER_TRIAL_FILES_TARGET,
            )
