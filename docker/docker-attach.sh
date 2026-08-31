#!/bin/bash

set -e
set -u

service="instinctlab"
if [ $# -eq 1 ]
then
    service=$1
fi

docker exec --interactive --tty "$service" bash
