#!/bin/bash
# Tag the local num-gpu:dev image with the current commit and push it to GitHub's registry.
# Needs: gh token with write:packages (gh auth refresh -s write:packages) and docker access.
# New GHCR packages are private by default.
set -euo pipefail
cd "$(dirname "$0")/.."
git diff --quiet HEAD -- julia scripts docker/Dockerfile docker/entry_help.sh NUMmodel/input || { echo "uncommitted changes: commit and rebuild first"; exit 1; }
tag=$(git rev-parse --short HEAD)
img=ghcr.io/georgehagstrom/num-gpu
# (old gh versions have no `gh auth token`; read the stored token from gh's config instead)
if gh auth token >/dev/null 2>&1; then token=$(gh auth token)
else token=$(sed -n 's/^ *oauth_token: *//p' ~/.config/gh/hosts.yml | head -1); fi
printf '%s' "$token" | docker login ghcr.io -u georgehagstrom --password-stdin
docker tag num-gpu:dev "$img:$tag"
docker tag num-gpu:dev "$img:latest"
docker push "$img:$tag"
docker push "$img:latest"
echo "pushed $img:$tag"
