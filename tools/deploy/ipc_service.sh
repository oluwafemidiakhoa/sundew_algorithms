#!/bin/bash
# Placeholder deployment script for installing Sundew IPC daemon.
# Usage: ./ipc_service.sh install
set -euo pipefail

COMMAND=${1:-help}
SERVICE_NAME=sundew-ipc.service
UNIT_PATH=/etc/systemd/system/$SERVICE_NAME

case "$COMMAND" in
  install)
    echo "(placeholder) Copy binaries and unit file to target"
    ;;
  remove)
    echo "(placeholder) Remove installed service"
    ;;
  *)
    echo "Usage: $0 {install|remove}" && exit 1
    ;;
esac
