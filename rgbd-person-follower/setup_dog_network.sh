#!/usr/bin/env bash
# Configure only the Jetson's dog Ethernet port; keep Wi-Fi and existing profiles.
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
[[ "$(nmcli -g connection.interface-name connection show "$profile")" == "$interface" ]]
[[ "$(nmcli -g ipv4.method connection show "$profile")" == manual ]]
[[ "$(nmcli -g ipv4.addresses connection show "$profile")" == 192.168.168.66/24 ]]
[[ "$(nmcli -g ipv4.never-default connection show "$profile")" == yes ]]
[[ -z "$(nmcli -g ipv4.gateway connection show "$profile")" ]]
[[ -z "$(nmcli -g ipv4.dns connection show "$profile")" ]]
echo "READY: $profile on $interface = 192.168.168.66/24, no default route. Plug in the dog's cable when requested."
