# IPC Daemon Deployment (Draft)

- Create systemd service templates per board
- Ship install script to copy bindings + daemon
- Configure network/firewall rules for gRPC/TCP ports
- Set up telemetry logging to persistent storage


Run `tools/deploy/ipc_service.sh install` on the board after editing copy paths; the default systemd unit is `tools/deploy/sundew-ipc.service` and starts your daemon script (replace ExecStart with actual command).
