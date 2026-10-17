#!/usr/bin/env bash
# Configure dog Ethernet and reserve its SDK port; keep Wi-Fi and existing profiles.
set -euo pipefail
interface=enP8p1s0
profile=dog-sdk
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --check ) ]]; then
  echo 'Usage: sudo bash setup_dog_network.sh | bash setup_dog_network.sh --check' >&2
  exit 2
fi
[[ -d "/sys/class/net/$interface" && ! -d "/sys/class/net/$interface/wireless" ]] || {
  echo "Expected Jetson Ethernet interface $interface is absent" >&2; exit 1;
}
if [[ "${1:-}" != --check ]]; then
  [[ $EUID -eq 0 ]] || { echo 'Run once with sudo; no Wi-Fi settings will be changed.' >&2; exit 1; }
  if nmcli -g connection.id connection show "$profile" >/dev/null 2>&1; then
    [[ "$(nmcli -g connection.interface-name connection show "$profile")" == "$interface" ]] || {
      echo "Refusing to modify an existing $profile profile for another interface" >&2; exit 1;
    }
  else
    nmcli connection add type ethernet ifname "$interface" con-name "$profile" \
      ipv4.method manual ipv4.addresses 192.168.168.66/24 ipv4.never-default yes \
      ipv4.ignore-auto-dns yes ipv6.method disabled connection.autoconnect yes \
      connection.autoconnect-priority 100
  fi
  nmcli connection modify "$profile" ipv4.method manual ipv4.addresses 192.168.168.66/24 \
    ipv4.gateway '' ipv4.dns '' ipv4.never-default yes ipv4.ignore-auto-dns yes \
    ipv6.method disabled connection.autoconnect yes connection.autoconnect-priority 100
fi
# Keep automatic ephemeral services from randomly taking the SDK's fixed port.
python3 - "${1:-}" <<'PY'
from pathlib import Path
import os, subprocess, sys
key = 'net.ipv4.ip_local_reserved_ports'
current = subprocess.check_output(['sysctl', '-n', key], text=True).strip()
ranges = [list(map(int, item.split('-'))) for item in current.split(',') if item]
reserved = any(bounds[0] <= 43988 <= bounds[-1] for bounds in ranges)
if sys.argv[1] == '--check':
    if not reserved:
        raise SystemExit('SDK port 43988 is not reserved; rerun this script with sudo.')
else:
    merged = current if reserved else ','.join(filter(None, (current, '43988')))
    path = Path('/etc/sysctl.d/99-z-rgbd-person-follower.conf')
    with path.open('w') as stream:
        stream.write(f'{key}={merged}\n')
        stream.flush()
        os.fsync(stream.fileno())
    subprocess.run(['sysctl', '-p', str(path)], check=True)
PY
if [[ "${1:-}" != --check ]]; then
  owner="$(ss -H -lunp 'sport = :43988')"
  if [[ "$owner" == *'("rpc.statd",'* ]]; then
    # Only restart the NFS status monitor; keep NFS service and exports running.
    systemctl restart rpc-statd.service
    systemctl is-active --quiet rpc-statd.service
  fi
  [[ -z "$(ss -H -lun 'sport = :43988')" ]] || {
    echo 'UDP 43988 is still occupied; its owner was left untouched.' >&2; exit 1;
  }
fi
[[ "$(nmcli -g connection.interface-name connection show "$profile")" == "$interface" ]]
[[ "$(nmcli -g ipv4.method connection show "$profile")" == manual ]]
[[ "$(nmcli -g ipv4.addresses connection show "$profile")" == 192.168.168.66/24 ]]
[[ "$(nmcli -g ipv4.never-default connection show "$profile")" == yes ]]
[[ -z "$(nmcli -g ipv4.gateway connection show "$profile")" ]]
[[ -z "$(nmcli -g ipv4.dns connection show "$profile")" ]]
echo "READY: $profile on $interface = 192.168.168.66/24, no default route; SDK port 43988 reserved."
