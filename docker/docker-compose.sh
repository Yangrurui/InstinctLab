#!/bin/bash

set -e
set -u

service="instinctlab"
if [ $# -eq 1 ]
then
    service=$1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_dir="$(cd "$script_dir/.." && pwd)"
build_python="${INSTINCTLAB_BUILD_PYTHON:-python3}"
export INSTINCTLAB_SOURCE_COMMIT
INSTINCTLAB_SOURCE_COMMIT="$(git -C "$repository_dir" rev-parse HEAD)"
"$build_python" "$repository_dir/scripts/build_release.py" \
    --output "$repository_dir/dist/release" \
    --expected-version 0.1.0
docker compose --file "$script_dir/docker-compose.yaml" --env-file "$script_dir/.env.base" build "$service"
docker compose --file "$script_dir/docker-compose.yaml" --env-file "$script_dir/.env.base" up --detach "$service"
