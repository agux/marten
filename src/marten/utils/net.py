import psutil
import socket

# Get all network interfaces and their IP addresses
def get_machine_ips():
    ip_addresses = []
    for interface, addrs in psutil.net_if_addrs().items():
        # Check if the interface is a physical NIC or WiFi device
        if interface.startswith(("eth", "en", "wlan", "wl")):
            for addr in addrs:
                if addr.family == socket.AF_INET:  # IPv4 addresses
                    ip_addresses.append(addr.address)
    return ip_addresses
