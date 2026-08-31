#!/bin/bash

set -e
set -u

service="instinctlab"
if [ $# -eq 1 ]
then
    service=$1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker compose --file "$script_dir/docker-compose.yaml" --env-file "$script_dir/.env.base" build "$service"
docker compose --file "$script_dir/docker-compose.yaml" --env-file "$script_dir/.env.base" up --detach "$service"
