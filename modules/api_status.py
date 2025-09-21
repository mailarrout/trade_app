# api_status.py
import logging
from PyQt5.QtCore import QTimer
import requests

# Get logger for this module
logger = logging.getLogger(__name__)

class ApiStatus:
    def __init__(self, main_window):
        self.main_window = main_window
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_status)
        self.is_online = False
        self.consecutive_failures = 0
        
    def start(self, interval=30000):  # 30 seconds
        """Start status monitoring"""
        self.timer.start(interval)
        self.check_status()  # Initial check
        logger.info("API status monitoring started")
        
    def check_status(self):
        """Check API status and update status bar"""
        try:
            response = requests.get("https://api.shoonya.com/", timeout=5)
            is_online = response.status_code < 400
            
            if is_online != self.is_online:
                self.is_online = is_online
                self.consecutive_failures = 0
                status = "🟢 ONLINE" if is_online else "🔴 OFFLINE"
                
                # Update status bar
                self.main_window.statusbar.showMessage(f"API Status: {status}")
                
                # Also update ApiStatusLabel if it exists
                if hasattr(self.main_window, 'ApiStatusLabel'):
                    self.main_window.ApiStatusLabel.setText("🟢" if is_online else "🔴")
                
                if is_online:
                    logger.info(f"API came ONLINE (response: {response.status_code})")
                else:
                    logger.warning(f"API went OFFLINE (status code: {response.status_code})")
                
            elif not is_online:
                # Track consecutive failures when offline
                self.consecutive_failures += 1
                if self.consecutive_failures >= 3:
                    logger.error(f"API has been offline for {self.consecutive_failures} consecutive checks")
                
        except requests.exceptions.Timeout:
            self._handle_offline("Timeout after 5 seconds")
        except requests.exceptions.ConnectionError:
            self._handle_offline("Connection error")
        except requests.exceptions.RequestException as e:
            self._handle_offline(f"Request error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in API status check: {e}")
            self._handle_offline(f"Unexpected error: {e}")

    def _handle_offline(self, error_message):
        """Handle offline status with consistent logging"""
        if self.is_online:
            self.is_online = False
            self.consecutive_failures = 1
            
            self.main_window.statusbar.showMessage("API Status: 🔴 OFFLINE")
            
            if hasattr(self.main_window, 'ApiStatusLabel'):
                self.main_window.ApiStatusLabel.setText("🔴")
            
            logger.warning(f"API went OFFLINE: {error_message}")
        else:
            # Increment consecutive failures
            self.consecutive_failures += 1
            if self.consecutive_failures >= 3:
                logger.error(f"API remains OFFLINE: {error_message} (consecutive failures: {self.consecutive_failures})")