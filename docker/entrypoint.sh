#!/bin/sh
set -e
mkdir -p /data
if [ "$(id -u)" -eq 0 ]; then
  chown cradle:cradle /data
  exec runuser -u cradle -g cradle -- "$@"
fi
exec "$@"
