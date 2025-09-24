# ===== main_app.py - Minimal Changes =====
import sys
import os
import logging
from datetime import datetime, timedelta
from pytz import timezone

# PyQt5
from PyQt5.QtWidgets import QApplication, QMessageBox, QPlainTextEdit
from PyQt5.QtCore import QTimer

# ===== CONFIGURATION FLAG =====
DEBUG_MODE = False  # True for detailed debugging
# ==============================

# ===== Logging Configuration =====
IST = timezone("Asia/Kolkata")

class ISTFormatter(logging.Formatter):
    """Custom formatter that converts timestamps to IST"""
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=IST)
        if datefmt:
            return dt.strftime(datefmt)
        else:
            return dt.strftime('%Y-%m-%d %H:%M:%S IST')

current_date = datetime.now(IST).strftime('%Y-%m-%d')
log_dir = os.path.join("logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"{current_date}_app.log")

log_level = logging.DEBUG if DEBUG_MODE else logging.INFO

# Clear any existing handlers
logging.getLogger().handlers = []

# Create logger
logger = logging.getLogger()
logger.setLevel(log_level)

# File handler (replaces basicConfig)
file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
ist_formatter = ISTFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(ist_formatter)
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(log_level if DEBUG_MODE else logging.INFO)
console_handler.setFormatter(ist_formatter)
logger.addHandler(console_handler)

logger.info(f"Application starting - Logging to {log_file} (Mode: {'DEBUG' if DEBUG_MODE else 'INFO'})")
logger.info(f"Logging timezone: IST (Asia/Kolkata)")

# ===== Import Modules =====

from modules.interface_loader import TradingUI
from modules.client_manager import ClientManager
from modules.position_manager import PositionManager
from modules.option_loader import OptionLoader
from modules.opstra_option_loader import OpstraOptionLoader
from modules.strategy_ibbm import IBBMStrategy
# from modules.strategy_ibbm_actual import IBBMStrategy
from modules.strategy_monthly_straddle import MonthlyStraddleStrategy
from modules.api_status import ApiStatus
from modules.price_chart import GraphPlotTab
from modules.payoff_graph import PayoffGraphTab

# ===== Custom UI Logging Handler =====
class QPlainTextEditHandler(logging.Handler):
    """Send logs to QPlainTextEdit"""
    def __init__(self, text_edit):
        super().__init__()
        self.text_edit = text_edit

    def emit(self, record):
        try:
            # Use IST formatter for UI logs too
            formatter = ISTFormatter("%(asctime)s - %(levelname)s - %(message)s")
            msg = formatter.format(record)
            self.text_edit.appendPlainText(msg)
        except Exception:
            pass

# ===== Main Application =====
class TradingApp:
    def __init__(self):
        try:
            logger.info("Initializing Trading Application")

            # --- QApplication + UI ---
            self.app = QApplication(sys.argv)
            self.ui = TradingUI()

            # --- UI logging ---
            self.setup_ui_logging()

            # --- Core Managers ---
            self.client_manager = ClientManager(self.ui)
            self.position_manager = PositionManager(self.ui, self.client_manager)
            self.option_loader = OptionLoader(self.ui)

            # --- Wire managers to UI ---
            self.ui.client_manager = self.client_manager
            self.ui.position_manager = self.position_manager
            self.ui.option_loader = self.option_loader
                        
            # --- Auto-load clients ---
            if hasattr(self.ui, "LoadClients") and hasattr(self.ui, "load_clients_clicked"):
                self.ui.LoadClients.clicked.connect(self.ui.load_clients_clicked)
                self.ui.load_clients_clicked()
                logger.info("Clients loaded successfully at startup")
            else:
                logger.warning("UI missing LoadClients or load_clients_clicked")

            self.ui.show()
            logger.info(f"UI shown at {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}")

            QTimer.singleShot(15000, self.recover_strategy_state)

            # --- Background Services ---
            self.api_status = ApiStatus(self.ui)
            self.api_status.start()

            self.opstra_loader = OpstraOptionLoader(self.ui, self.client_manager)
            if hasattr(self.ui, "LoadOpstraPushButton"):
                self.ui.LoadOpstraPushButton.clicked.connect(self.opstra_loader.load_opstra_data)
            self.ui.opstra_loader = self.opstra_loader

            # --- Graph Tabs ---
            self.graph_tab = GraphPlotTab(self.ui, self.client_manager)
            self.payoff_tab = PayoffGraphTab(self.ui, self.client_manager)
            QTimer.singleShot(5000, self.payoff_tab.calculate_adjustment_points)

            # --- Strategies ---
            self.strategies = {
                "IBBM Intraday": IBBMStrategy(self.ui, self.client_manager),
                "Monthly Straddle": MonthlyStraddleStrategy(self.ui, self.client_manager, self.position_manager,"NFO_symbols.txt"),
            }

            self.current_strategy = "IBBM Intraday"
            self.enable_ibbm_auto_run = True

            # Connect UI signals for strategy selection
            if hasattr(self.ui, "ExecuteStrategyQPushButton") and hasattr(self.ui, "StrategyNameQComboBox"):
                self.ui.ExecuteStrategyQPushButton.clicked.connect(self.execute_selected_strategy)
                self.ui.StrategyNameQComboBox.currentTextChanged.connect(self.on_strategy_changed)
                self.ui.CurrentStrategyQLabel.setText(f"Current: {self.current_strategy}")

            # Auto-run IBBM
            if self.enable_ibbm_auto_run:
                self.start_strategy("IBBM Intraday")

            # --- Auto-close at 15:31 IST ---
            now_ist = datetime.now(IST)
            target_time = now_ist.replace(hour=15, minute=31, second=0, microsecond=0)
            if now_ist >= target_time:
                target_time += timedelta(days=1)
            delay_ms = int((target_time - now_ist).total_seconds() * 1000)
            QTimer.singleShot(delay_ms, self.cleanup)
            logger.info(f"Application will auto-close at {target_time.strftime('%Y-%m-%d %H:%M:%S IST')}")

            logger.info("Trading application initialized successfully")

        except Exception as e:
            logger.critical(f"Failed to initialize trading application: {e}")
            raise

    # ===== UI Logging =====
    def setup_ui_logging(self):
        """Setup logging to QPlainTextEdit in UI"""
        try:
            logs_widget = self.ui.findChild(QPlainTextEdit, "LogsQPlainText")
            if logs_widget:
                ui_handler = QPlainTextEditHandler(logs_widget)
                formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
                ui_handler.setFormatter(formatter)
                ui_handler.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)
                logging.getLogger().addHandler(ui_handler)
                logger.info("UI logging initialized (QPlainTextEdit)")
            else:
                logger.warning("LogsQPlainText widget not found in UI")
        except Exception as e:
            logger.error(f"Failed to setup UI logging: {e}")

    # ===== Strategy Handlers =====
    def on_strategy_changed(self, strategy_name):
        """Update current strategy"""
        self.current_strategy = strategy_name
        self.ui.CurrentStrategyQLabel.setText(f"Current: {strategy_name}")

    def execute_selected_strategy(self):
        """Execute the selected strategy from UI"""
        try:
            strategy_name = self.ui.StrategyNameQComboBox.currentText()
            logger.info(f"Executing strategy: {strategy_name}")
            self.current_strategy = strategy_name

            self.start_strategy(strategy_name)

        except Exception as e:
            logger.error(f"Error executing strategy: {e}")
            QMessageBox.critical(self.ui, "Error", f"Failed to execute strategy: {e}")

    def start_strategy(self, strategy_name):
        """Start or run a strategy"""
        strategy_obj = self.strategies.get(strategy_name)
        if strategy_obj:
            # Start IBBM timers if not running
            if strategy_name == "IBBM Intraday" and self.enable_ibbm_auto_run:
                for timer_attr in ["strategy_timer", "monitor_timer"]:
                    timer = getattr(strategy_obj, timer_attr, None)
                    if timer and not timer.isActive():
                        interval_attr = "STRATEGY_CHECK_INTERVAL" if timer_attr == "strategy_timer" else "MONITORING_INTERVAL"
                        timer.start(getattr(strategy_obj, interval_attr))

            # Execute strategy method
            if hasattr(strategy_obj, "on_execute_strategy_clicked"):
                strategy_obj.on_execute_strategy_clicked()
            elif hasattr(strategy_obj, "execute_strategy"):
                strategy_obj.execute_strategy()
            else:
                logger.warning(f"No execution method for {strategy_name}")
        else:
            logger.warning(f"Strategy not implemented: {strategy_name}")
            QMessageBox.information(self.ui, "Info", f"{strategy_name} strategy not implemented")

    # ===== Cleanup =====
    def cleanup(self):
        """Stop timers, logout clients, cleanup resources"""
        try:
            logger.info("Cleaning up application resources")

            # Stop API monitoring
            if hasattr(self, "api_status"):
                self.api_status.stop_monitoring()

            # Stop strategy timers
            for strategy_obj in self.strategies.values():
                for timer_attr in ["strategy_timer", "monitor_timer"]:
                    timer = getattr(strategy_obj, timer_attr, None)
                    if timer and timer.isActive():
                        timer.stop()

            # Stop graph updates
            if hasattr(self, "graph_tab"):
                self.graph_tab.stop_updates()
            if hasattr(self, "payoff_tab"):
                self.payoff_tab.stop_updates()

            # Stop position manager
            if hasattr(self, "position_manager"):
                self.position_manager.stop_updates()

            # Logout clients
            if hasattr(self, "client_manager"):
                for client_name, client_id, client in getattr(self.client_manager, "clients", []):
                    try:
                        client.logout()
                    except Exception:
                        pass

            logger.info("Cleanup completed. Exiting application.")
            self.app.quit()

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            self.app.quit()

    def recover_strategy_state(self):
        """Check current positions and re-activate the corresponding strategies."""
        try:
            logger.info("Attempting to recover strategy state from existing positions.")
            
            # First try to recover from state files (new preferred method)
            recovery_success = False
            for strategy_name, strategy_obj in self.strategies.items():
                logger.info(f"Attempting to recover {strategy_name} from state file...")
                
                if hasattr(strategy_obj, 'recover_from_state_file'):
                    try:
                        if strategy_obj.recover_from_state_file():
                            logger.info(f"Successfully recovered {strategy_name} from state file")
                            recovery_success = True
                            
                            # Set UI to reflect recovered strategy
                            if hasattr(self.ui, 'StrategyNameQComboBox'):
                                index = self.ui.StrategyNameQComboBox.findText(strategy_name)
                                if index >= 0:
                                    self.ui.StrategyNameQComboBox.setCurrentIndex(index)
                                    self.current_strategy = strategy_name
                                    self.ui.CurrentStrategyQLabel.setText(f"Current: {strategy_name} (Recovered)")
                                    logger.info(f"UI updated for recovered strategy: {strategy_name}")
                            
                            # Break after first successful recovery (assuming one active strategy at a time)
                            break
                        else:
                            logger.info(f"State file recovery failed for {strategy_name}")
                    except Exception as e:
                        logger.error(f"Error during state file recovery for {strategy_name}: {e}")
            
            # If state file recovery succeeded, we're done
            if recovery_success:
                logger.info("Strategy recovery completed successfully from state files")
                return
            
            # Fallback to position-based recovery if state file recovery fails or no state files found
            logger.info("Falling back to position-based recovery...")
            strategy_assignments = self.position_manager.get_all_strategy_assignments()
            
            if not strategy_assignments:
                logger.info("No active strategy assignments found. Starting from scratch.")
                return

            # Group assignments by strategy name
            from collections import defaultdict
            strategy_positions_map = defaultdict(list)
            
            for assignment in strategy_assignments:
                # Unpack all values including the new entry_spot_price
                strategy_name, symbol, token, net_qty, avg_price, entry_spot_price = assignment
                
                strategy_positions_map[strategy_name].append({
                    "symbol": symbol, 
                    "token": token, 
                    "net_qty": net_qty, 
                    "avg_price": avg_price,
                    "entry_spot_price": entry_spot_price
                })

            logger.info(f"Strategy Positions Map: {dict(strategy_positions_map)}")

            # For each strategy that has active positions, activate it and pass the positions
            for strategy_name, positions_list in strategy_positions_map.items():
                strategy_obj = self.strategies.get(strategy_name)
                if strategy_obj:
                    logger.info(f"Found strategy object for {strategy_name}. Attempting to recover state.")
                    
                    # Check if the strategy has a method to handle state recovery
                    if hasattr(strategy_obj, 'recover_from_positions'):
                        # Call the method and pass the enhanced positions list
                        success = strategy_obj.recover_from_positions(positions_list)
                        if success:
                            logger.info(f"Successfully recovered state for {strategy_name}")
                            
                            # Set the UI dropdown to reflect the active strategy
                            if hasattr(self.ui, 'StrategyNameQComboBox'):
                                index = self.ui.StrategyNameQComboBox.findText(strategy_name)
                                if index >= 0:
                                    self.ui.StrategyNameQComboBox.setCurrentIndex(index)
                                    self.current_strategy = strategy_name
                                    self.ui.CurrentStrategyQLabel.setText(f"Current: {strategy_name} (Recovered)")
                        else:
                            logger.error(f"Failed to recover state for {strategy_name}")
                    else:
                        logger.warning(f"Strategy {strategy_name} does not have a 'recover_from_positions' method.")
                else:
                    logger.warning(f"Could not find strategy object for: {strategy_name}")

        except Exception as e:
            logger.error(f"Error in recover_strategy_state: {e}", exc_info=True)

    # ===== Run =====
    def run(self):
        try:
            logger.info(f"Application running at {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}")
            result = self.app.exec_()
            logger.info("Application exited normally")
            sys.exit(result)
        except Exception as e:
            logger.critical(f"Crash during execution: {e}")
            sys.exit(1)

# ===== Entry Point =====
def main():
    try:
        app = TradingApp()
        app.run()
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        QMessageBox.critical(None, "Fatal Error", f"Application failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()