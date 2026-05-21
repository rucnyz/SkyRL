#!/usr/bin/env bash
# Build + push the 6 per-skill base images for Nemotron-Terminal-Synthetic-Tasks
# (mixed/ subset) to ghcr.io. Each task's environment/Dockerfile ends with
# `COPY files/ /app/` to bake in per-task input data; we strip that so the
# image is task-agnostic and 5984 tasks share one of 6 skill-base images.
# At runtime, harbor_pgc/environments/skyrl_e2b.SharedTemplateE2BEnvironment
# uploads per-task files/ into /app/ after sandbox.create.
#
# Usage:
#   bash examples/train_integrations/harbor_pgc/scripts/build_skill_images.sh
#
# Requires:
#   - docker daemon reachable on $DOCKER_HOST (or default /var/run/docker.sock)
#   - `gh auth token` returns a PAT with `write:packages`
#   - dataset already prepared via prepare_harbor_dataset.py
#
# One-time post-step (no API for user-owned packages): flip each package's
# visibility to public at
#   https://github.com/users/${GHCR_OWNER}/packages/container/<name>/settings
# (set "Inherit access from source repository" once the source label points
# at a public repo, or toggle "Change visibility" → Public).

set -euo pipefail

GHCR_OWNER="${GHCR_OWNER:-rucnyz}"
IMAGE_TAG="${IMAGE_TAG:-1.0}"
SKYRL_REPO_URL="${SKYRL_REPO_URL:-https://github.com/${GHCR_OWNER}/SkyRL}"

DATASET_DIR="${DATASET_DIR:-$(dirname "$0")/../data/Nemotron-Terminal-Synthetic-Tasks}"
DATASET_DIR="$(cd "$DATASET_DIR" && pwd)"
SKILL_BASED="${DATASET_DIR}/skill_based"

if [ ! -d "$SKILL_BASED" ]; then
  echo "skill_based/ not found at: $SKILL_BASED" >&2
  echo "Run: uv run examples/train_integrations/harbor_pgc/prepare_harbor_dataset.py" >&2
  exit 1
fi

# Authenticate to ghcr.io with the gh CLI's PAT (no-op if already logged in).
if ! docker info 2>/dev/null | grep -q "ghcr.io" 2>/dev/null; then
  echo "Logging in to ghcr.io..."
  gh auth token | docker login ghcr.io -u "$GHCR_OWNER" --password-stdin
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# All 11 skills shipping in Nemotron-Terminal-Synthetic-Tasks across all
# three shard layouts (mixed has 6, easy adds 3, medium adds 2 more). The
# Dockerfile for any skill is identical across the shards that include it,
# so we just need to find ONE tarball that has the skill.
SKILLS=(
  data_processing
  data_science
  debugging
  file_operations
  scientific_computing
  security
  data_querying
  software_engineering
  dependency_management
  model_training
  system_administration
)

# Tarball search order. mixed/<skill>.tar.gz first (smallest, fastest to
# list), then the broader shards as fallbacks.
locate_dockerfile_src() {
  local skill="$1"
  for cand in \
    "$SKILL_BASED/mixed/${skill}.tar.gz" \
    "$SKILL_BASED/easy.tar.gz" \
    "$SKILL_BASED/medium_shard1.tar.gz" \
    "$SKILL_BASED/medium_shard2.tar.gz"; do
    [ -f "$cand" ] || continue
    local task_path
    task_path="$(tar -tzf "$cand" | awk -F/ -v s="$skill" '$0 ~ ("/" s "/") && /\/task.toml$/ {sub(/\/task.toml$/, ""); print; exit}')"
    if [ -n "$task_path" ]; then
      echo "$cand|$task_path"
      return 0
    fi
  done
  return 1
}

for skill in "${SKILLS[@]}"; do
  echo "================================================================"
  echo "skill: $skill"
  echo "================================================================"

  found="$(locate_dockerfile_src "$skill")" || {
    echo "  skip $skill (no tarball contains it)" >&2
    continue
  }
  src="${found%%|*}"
  task_path="${found##*|}"
  echo "  source: $(basename "$src") :: $task_path"
  task_skill_dir="$WORK/${skill}"
  rm -rf "$task_skill_dir"
  mkdir -p "$task_skill_dir"
  tar -xzf "$src" -C "$task_skill_dir" "$task_path/environment/Dockerfile"
  src_dockerfile="$task_skill_dir/$task_path/environment/Dockerfile"

  # Generate the build-context Dockerfile in one awk pass:
  #   - strip `COPY files/ /app/` (the per-task data layer; we upload it at
  #     runtime via SharedTemplateE2BEnvironment)
  #   - inject OCI metadata labels right after FROM
  build_ctx="$WORK/build_${skill}"
  rm -rf "$build_ctx" && mkdir -p "$build_ctx"
  awk -v src="$SKYRL_REPO_URL" -v desc="PGC harbor_pgc $skill skill base (Nemotron-Terminal-Synthetic-Tasks). COPY files/ stripped — supply per-trial via harbor_pgc/environments/skyrl_e2b." '
    /^FROM /{
      print; print "";
      print "LABEL org.opencontainers.image.source=\"" src "\"";
      print "LABEL org.opencontainers.image.description=\"" desc "\"";
      print "LABEL org.opencontainers.image.licenses=\"Apache-2.0\"";
      next
    }
    /^COPY files\// {next}
    {print}
  ' "$src_dockerfile" > "$build_ctx/Dockerfile"

  img="ghcr.io/${GHCR_OWNER}/nemotron-${skill}:${IMAGE_TAG}"
  echo "Building $img"
  ( cd "$build_ctx" && docker build -t "$img" . )
  echo "Pushing $img"
  docker push "$img"
done

echo
echo "Done. Images:"
for skill in "${SKILLS[@]}"; do
  echo "  ghcr.io/${GHCR_OWNER}/nemotron-${skill}:${IMAGE_TAG}"
done
echo
echo "Next: flip each package's visibility to public via the GitHub UI:"
echo "  https://github.com/users/${GHCR_OWNER}?tab=packages"
echo "(Once linked to the public SkyRL repo via the source label, the package"
echo " can be toggled to inherit access from the source repo's visibility.)"
