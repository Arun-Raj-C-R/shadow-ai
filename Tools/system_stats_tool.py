# tools/system_stats_tool.py
import psutil
import platform
from datetime import datetime

def get_system_stats() -> str:
    """
    Returns a comprehensive, human-readable summary of system stats:
    CPU, RAM, GPU (if available), Disk, Network, Temp, Battery, and Uptime.
    """
    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_freq = psutil.cpu_freq()
    cpu_cores = psutil.cpu_count(logical=False)
    cpu_threads = psutil.cpu_count(logical=True)

    # Memory
    ram = psutil.virtual_memory()
    swap = psutil.swap_memory()

    # Disk
    disk = psutil.disk_usage('/')
    disk_io = psutil.disk_io_counters()

    # Network
    net_io = psutil.net_io_counters()

    # Uptime
    uptime_sec = int((datetime.now() - datetime.fromtimestamp(psutil.boot_time())).total_seconds())
    hours, rem = divmod(uptime_sec, 3600)
    minutes, seconds = divmod(rem, 60)

    # Battery
    battery = psutil.sensors_battery()
    battery_info = (
        f"{battery.percent:.0f}% ({'Charging' if battery.power_plugged else 'Discharging'})"
        if battery else "No battery detected"
    )

    # Temperatures (if supported)
    temps = psutil.sensors_temperatures() if hasattr(psutil, "sensors_temperatures") else {}
    cpu_temp = None
    if temps:
        for name, entries in temps.items():
            for entry in entries:
                if "cpu" in name.lower() or "core" in entry.label.lower():
                    cpu_temp = entry.current
                    break
            if cpu_temp:
                break
    cpu_temp_str = f"{cpu_temp:.1f}°C" if cpu_temp else "N/A"

    # System info
    system_info = f"{platform.system()} {platform.release()} ({platform.machine()})"

    # Compile summary
    return (
        f"=== SYSTEM STATUS ===\n"
        f"System: {system_info}\n"
        f"Uptime: {hours}h {minutes}m {seconds}s\n"
        f"\n"
        f"CPU: {cpu_percent:.1f}%  |  {cpu_cores} cores / {cpu_threads} threads  |  {cpu_freq.current:.0f} MHz\n"
        f"CPU Temp: {cpu_temp_str}\n"
        f"RAM: {ram.percent:.1f}% ({ram.used // (1024**3)} GB / {ram.total // (1024**3)} GB)\n"
        f"Swap: {swap.percent:.1f}% ({swap.used // (1024**3)} GB / {swap.total // (1024**3)} GB)\n"
        f"Disk: {disk.percent:.1f}% ({disk.used // (1024**3)} GB / {disk.total // (1024**3)} GB)\n"
        f"Disk I/O: R {disk_io.read_bytes // (1024**2)} MB | W {disk_io.write_bytes // (1024**2)} MB\n"
        f"Network I/O: Sent {net_io.bytes_sent // (1024**2)} MB | Received {net_io.bytes_recv // (1024**2)} MB\n"
        f"Battery: {battery_info}\n"
        f"======================"
    )

# Example usage
if __name__ == "__main__":
    print(get_system_stats())
