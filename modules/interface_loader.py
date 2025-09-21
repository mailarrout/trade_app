from PyQt5.QtWidgets import QMainWindow, QPushButton, QMessageBox
from PyQt5 import uic
from PyQt5.QtCore import QTimer
from datetime import datetime 
from pytz import timezone
import logging

# Get logger for this module
logger = logging.getLogger(__name__)

IST = timezone('Asia/Kolkata')

class TradingUI(QMainWindow):
    def __init__(self):
        super().__init__()
        try:
            start_time = datetime.now(IST)
            logger.info("Loading UI from Dashboard.ui file")
            
            uic.loadUi("Dashboard.ui", self)
            logger.info(f"UI loaded successfully at {start_time.strftime('%Y-%m-%d %H:%M:%S IST')}")

            self.client_manager = None
            self.position_manager = None
            self.option_loader = None
            self.graph_tab = None
            self.opstra_loader = None

            # UI Connections
            logger.debug("Setting up UI button connections")
            self.LoadClients.clicked.connect(self.load_clients_clicked)
            
            # Setup tables
            self._setup_tables()
            
            load_time = (datetime.now(IST) - start_time).total_seconds()
            logger.debug(f"UI initialization completed in {load_time:.2f} seconds")
                        
        except FileNotFoundError:
            error_msg = "Dashboard.ui file not found"
            logger.critical(error_msg)
            QMessageBox.critical(self, "Error", error_msg)
            raise
        except Exception as e:
            error_msg = f"UI initialization failed at {datetime.now(IST).strftime('%H:%M:%S IST')}: {str(e)}"
            logger.error(error_msg)
            QMessageBox.critical(self, "Error", f"UI initialization failed: {str(e)}")
            raise

    def load_clients_clicked(self):
        """Handle client loading button click"""
        try:
            start_time = datetime.now(IST)
            logger.info(f"Client loading initiated at {start_time.strftime('%H:%M:%S IST')}")
            
            if self.client_manager and self.position_manager:
                logger.debug("Both client_manager and position_manager are available")
                
                if self.client_manager.load_clients():
                    logger.info("Client loading successful, starting position updates")
                    self.position_manager.start_updates()
                    logger.info("Clients, positions and MTM loaded successfully")
                    
                    # Load expiry dates with timing
                    expiry_start = datetime.now(IST)
                    logger.info("Loading expiry dates...")
                    self.client_manager.load_expiry_dates()
                    expiry_time = (datetime.now(IST) - expiry_start).total_seconds()
                    logger.info(f"Expiry dates loaded in {expiry_time:.2f} seconds")
                else:
                    logger.error("Client loading failed - load_clients() returned False")
                    QMessageBox.warning(self, "Warning", "Client loading failed. Check ClientInfo.txt file.")
            else:
                error_msg = "Manager(s) not initialized when loading clients"
                if not self.client_manager:
                    error_msg += " - client_manager is None"
                if not self.position_manager:
                    error_msg += " - position_manager is None"
                logger.error(error_msg)
                QMessageBox.warning(self, "Warning", "System not fully initialized. Please restart the application.")
                
            total_time = (datetime.now(IST) - start_time).total_seconds()
            logger.info(f"Client loading process completed in {total_time:.2f} seconds")
            
        except Exception as e:
            error_msg = f"Error in load_clients_clicked at {datetime.now(IST).strftime('%H:%M:%S IST')}: {str(e)}"
            logger.error(error_msg)
            QMessageBox.critical(self, "Error", f"Failed to load clients: {str(e)}")

    def _setup_tables(self):
        """Initialize tables with headers"""
        try:
            start_time = datetime.now(IST)
            logger.debug("Setting up table headers and column widths")
            
            # Position Table
            pos_headers = [
                "Symbol", "PE/CE", "Buy Q", "Sell Q", "Net Q",
                "Sell P", "Buy P", "LTP", "MTM", "PnL", "Product", 
                "Strategy", "Action"
            ]
            widths = [200, 60, 80, 80, 80, 60, 60, 60, 80, 80, 80, 120, 80]
            
            self.PositionTable.setColumnCount(len(pos_headers))
            self.PositionTable.setHorizontalHeaderLabels(pos_headers)
            for i, width in enumerate(widths):
                self.PositionTable.setColumnWidth(i, width)
            
            logger.debug(f"Position table setup with {len(pos_headers)} columns")

            # Clients Table
            client_headers = ["Client", "MTM", "P&L"]
            widths = [200, 80, 80]
            
            self.AllClientsTable.setColumnCount(len(client_headers))
            self.AllClientsTable.setHorizontalHeaderLabels(client_headers)
            for i, width in enumerate(widths):
                self.AllClientsTable.setColumnWidth(i, width)
            
            logger.debug(f"Clients table setup with {len(client_headers)} columns")
                
            setup_time = (datetime.now(IST) - start_time).total_seconds()
            logger.debug(f"Tables initialized in {setup_time:.2f} seconds")
            
        except AttributeError as e:
            logger.error(f"Table setup failed - missing UI element: {str(e)}")
            # Continue without tables - this might be expected in some UI versions
        except Exception as e:
            logger.error(f"Error setting up tables at {datetime.now(IST).strftime('%H:%M:%S IST')}: {str(e)}")
            # Don't raise here as table setup failure shouldn't crash the whole app

    def log_message(self, category, message):
        """
        Add a message to the log window with IST timestamp
        Note: This is now only for UI display - actual logging is handled by app_logger
        """
        try:
            timestamp = datetime.now(IST).strftime("%H:%M:%S IST")
            log_entry = f"[{timestamp}] {category}: {message}"
            
            # Check if log widget exists
            if hasattr(self, 'LogsQPlainText'):
                self.LogsQPlainText.appendPlainText(log_entry)
                self.LogsQPlainText.ensureCursorVisible()
                
                # Auto-scroll to bottom if too many messages
                if self.LogsQPlainText.blockCount() > 1000:
                    self.LogsQPlainText.clear()
                    clear_msg = f"[{timestamp}] SYSTEM: Cleared log buffer"
                    self.LogsQPlainText.appendPlainText(clear_msg)
            else:
                logger.warning("LogsQPlainText widget not found in UI")
                
        except Exception as e:
            logger.warning(f"Failed to display log in UI: {str(e)}")