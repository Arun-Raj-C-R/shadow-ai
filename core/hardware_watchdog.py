import csv, time, os, pyttsx3, threading
from typing import Dict, Any, Optional, List

class HardwareWatchdog:
    """
    Monitors a Libre Hardware Monitor CSV file for overload conditions,
    stores the alert message for the main Shadow process to read, and 
    provides a full system status report on demand.
    """
    
    # NOTE: You MUST configure LHM to output to this path.
    CSV_PATH = r"C:\Shadow\stats.csv"

    def __init__(self, thresholds: Dict[str, float], cooldown: int = 30):
        self.thresholds = thresholds
        self.cooldown = cooldown
        self.last_alert: Dict[str, float] = {}
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # New variable to hold the latest alert message for Shadow to poll
        self.latest_alert_message: Optional[str] = None 
        
        # --- Voice Setup (Optional, but included for completeness) ---
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 170)
        except Exception:
            logging.warning("pyttsx3 failed to initialize. Voice alerts disabled.")
            self.engine = None
        
    def _speak(self, text: str):
        """Offline text-to-speech implementation (runs in separate thread)."""
        if self.engine:
            print(f"[SHADOW VOICE ALERT] {text}")
            self.engine.say(text)
            self.engine.runAndWait()

    def _read_latest_data(self) -> Dict[str, Any]:
        """Reads the header and latest data row from the CSV."""
        if not os.path.exists(self.CSV_PATH):
            return {"status": "CSV_NOT_FOUND"}
            
        try:
            with open(self.CSV_PATH, 'r') as f:
                rows = list(csv.reader(f))
                if len(rows) < 2:  # Need header and at least one data row
                    return {"status": "NO_DATA"}
                    
                header = rows[0]
                latest_data = dict(zip(header, rows[-1]))
                
                # Clean and convert the numeric metrics for direct use
                cleaned_data = {}
                for name, val_str in latest_data.items():
                    if val_str:
                        try:
                            # Strip units (% or Â°C) and convert to float
                            val = float(val_str.replace('%', '').replace('Â°C', '').strip())
                            cleaned_data[name] = val
                        except ValueError:
                            # Keep non-numeric data (like time stamps) as is
                            cleaned_data[name] = val_str
                
                return cleaned_data
            
        except Exception as e:
            logging.error(f"Error reading CSV file: {e}")
            return {"status": f"READ_ERROR: {e}"}

    def _monitor_loop(self):
        """The core thread for continuous overload detection."""
        while self.running:
            latest = self._read_latest_data()
            now = time.time()
            
            # --- Alert Detection Logic ---
            for name, thresh in self.thresholds.items():
                if name in latest and isinstance(latest[name], float):
                    val = latest[name]

                    if val > thresh:
                        if now - self.last_alert.get(name, 0) > self.cooldown:
                            self.last_alert[name] = now
                            
                            # **CRITICAL FIX: Store alert message for main Shadow thread**
                            alert_msg = f"SYSTEM OVERLOAD: {name} is at {val:.1f}, exceeding {thresh}."
                            self.latest_alert_message = alert_msg
                            
                            # Optional: Still provide voice feedback 
                            threading.Thread(target=self._speak, args=(alert_msg,), daemon=True).start()
                            
            time.sleep(1)

    # --- PUBLIC METHODS FOR SHADOW BRAIN ---

    def start(self):
        """Starts the watchdog monitoring thread."""
        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            self._speak("Shadow watchdog online.")

    def stop(self):
        """Stops the watchdog monitoring thread."""
        if self.running:
            self.running = False
            logging.info("Hardware Watchdog terminated.")

    def check_for_new_alert(self) -> Optional[str]:
        """
        *** PEAK CAPABILITY (ALERT PUSH) ***
        Returns the latest *critical* alert message and clears the flag. 
        The main Shadow loop MUST call this periodically.
        """
        alert = self.latest_alert_message
        self.latest_alert_message = None  # Clear flag after reading
        return alert

    def get_current_status_report(self) -> Dict[str, Any]:
        """
        *** PEAK CAPABILITY (STATUS PULL) ***
        Returns a full, cleaned dictionary of all current system metrics.
        The main Shadow loop calls this when asked for "status vulnerability".
        """
        report = self._read_latest_data()
        
        vulnerabilities = []
        for name, thresh in self.thresholds.items():
            if name in report and isinstance(report[name], float):
                val = report[name]
                if val > thresh * 0.85: # Check if within 15% of failure
                    vulnerabilities.append(f"HIGH {name}: {val:.1f} (Threshold: {thresh})")
                elif val > thresh * 0.5: # Check if usage is significant
                    vulnerabilities.append(f"MODERATE {name}: {val:.1f}")

        report['vulnerability_summary'] = vulnerabilities
        return report

# --- Example of Shadow Brain Polling ---
if __name__ == '__main__':
    WATCHDOG_THRESHOLDS = {
        "CPU Total": 85,  # % load
        "CPU Temp": 80,   # Â°C
        "RAM Used": 90,   # % used
    }
    
    # Initialize the tool
    watchdog = HardwareWatchdog(thresholds=WATCHDOG_THRESHOLDS)
    watchdog.start()
    
    print("\n[MAIN SHADOW LOOP STARTED]")
    
    # Simulate the main Shadow loop (which runs indefinitely)
    for i in range(15):
        time.sleep(2)
        
        # 1. SHADOW POLING FOR ALERT (Request 1)
        alert = watchdog.check_for_new_alert()
        if alert:
            # When an alert is received, Shadow can interrupt any task
            print(f"\n<<< SHADOW INTERRUPT: {alert} >>>")
            
        # 2. SHADOW PULLING FOR STATUS (Request 2)
        if i == 10:
            print("\nUser: 'Shadow, give me a system status vulnerability report.'")
            report = watchdog.get_current_status_report()
            
            # Shadow processes the report (example output format)
            cpu_load = report.get('CPU Total', 'N/A')
            cpu_temp = report.get('CPU Temp', 'N/A')
            ram_used = report.get('RAM Used', 'N/A')
            
            print("Shadow: Current System Status (Vulnerability Analysis)")
            print(f"  - CPU Load: {cpu_load}% | Temp: {cpu_temp}Â°C")
            if report['vulnerability_summary']:
                print("  - VULNERABILITIES DETECTED:")
                for v in report['vulnerability_summary']:
                    print(f"    * {v}")
            else:
                print("  - Status: Nominal.")
            
    watchdog.stop()
