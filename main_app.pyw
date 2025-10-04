# ===== main_app.pyw - Coordinator Version (Modified) =====
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

# ===== STRATEGY CONFIGURATION =====
# Choose default strategy from: 
# "IBBM Intraday", "Intraday Straddle", "Monthly Straddle"
DEFAULT_STRATEGY = "IBBM Intraday"
ENABLE_AUTO_RUN = True  # Set to False if you don't want IBBM auto-execution

# ADD THESE NEW FLAGS TO CONTROL STRATEGY AUTO-RUN
ENABLE_INTRADAY_STRADDLE_AUTO_RUN = False  # Set to False to prevent auto-run
ENABLE_MONTHLY_STRADDLE_AUTO_RUN = False   # Set to False to prevent auto-run
# ==============================

# ===== Logging Configuration =====
IST = timezone("Asia/Kolkata")

class ISTFormatter(logging.Formatter):
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

logging.getLogger().handlers = []

logger = logging.getLogger()
logger.setLevel(log_level)

file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
ist_formatter = ISTFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(ist_formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(log_level if DEBUG_MODE else logging.INFO)
console_handler.setFormatter(ist_formatter)
logger.addHandler(console_handler)

logger.info(f"Application starting - Logging to {log_file} (Mode: {'DEBUG' if DEBUG_MODE else 'INFO'})")
logger.info(f"Logging timezone: IST (Asia/Kolkata)")

# ===== Import Modules =====
try:
    from modules.interface_loader import TradingUI
    logger.info("Successfully imported TradingUI")
except Exception as e:
    logger.error(f"Failed to import TradingUI: {e}")
    sys.exit(1)

try:
    from modules.client_manager import ClientManager
    logger.info("Successfully imported ClientManager")
except Exception as e:
    logger.error(f"Failed to import ClientManager: {e}")

try:
    from modules.position_manager import PositionManager
    logger.info("Successfully imported PositionManager")
except Exception as e:
    logger.error(f"Failed to import PositionManager: {e}")

try:
    from modules.option_loader import OptionLoader
    logger.info("Successfully imported OptionLoader")
except Exception as e:
    logger.error(f"Failed to import OptionLoader: {e}")

try:
    from modules.opstra_option_loader import OpstraOptionLoader
    logger.info("Successfully imported OpstraOptionLoader")
except Exception as e:
    logger.error(f"Failed to import OpstraOptionLoader: {e}")

try:
    from modules.strategy_ibbm import IBBMStrategy
    # from modules.strategy_ibbm_actual import IBBMStrategy
    logger.info("Successfully imported IBBMStrategy")
except Exception as e:
    logger.error(f"Failed to import IBBMStrategy: {e}")


try:
    from modules.strategy_intraday_straddle import IntradayStraddleStrategy
    logger.info("Successfully imported IntradayStraddleStrategy")
except Exception as e:
    logger.error(f"Failed to import IntradayStraddleStrategy: {e}")

try:
    from modules.strategy_monthly_straddle import MonthlyStraddleStrategy
    logger.info("Successfully imported MonthlyStraddleStrategy")
except Exception as e:
    logger.error(f"Failed to import MonthlyStraddleStrategy: {e}")


try:
    from modules.api_status import ApiStatus
    logger.info("Successfully imported ApiStatus")
except Exception as e:
    logger.error(f"Failed to import ApiStatus: {e}")

try:
    from modules.price_chart import GraphPlotTab
    logger.info("Successfully imported GraphPlotTab")
except Exception as e:
    logger.error(f"Failed to import GraphPlotTab: {e}")

try:
    from modules.payoff_graph import PayoffGraphTab
    logger.info("Successfully imported PayoffGraphTab")
except Exception as e:
    logger.error(f"Failed to import PayoffGraphTab: {e}")

# ===== Custom UI Logging Handler =====
class QPlainTextEditHandler(logging.Handler):
    def __init__(self, text_edit):
        super().__init__()
        self.text_edit = text_edit

    def emit(self, record):
        try:
            formatter = ISTFormatter("%(asctime)s - %(levelname)s - %(message)s")
            msg = formatter.format(record)
            self.text_edit.appendPlainText(msg)
        except Exception as e:
            logger.debug(f"UI logging emit failed: {e}")

# ===== Main Application =====
class TradingApp:
    def __init__(self):
        try:
            logger.info("=== INITIALIZING TRADING APPLICATION ===")

            # --- QApplication + UI ---
            logger.info("Creating QApplication instance")
            self.app = QApplication(sys.argv)
            
            logger.info("Loading Trading UI")
            self.ui = TradingUI()
            logger.info("Trading UI loaded successfully")

            self.ui.show()

            # --- UI logging ---
            self.setup_ui_logging()

            # --- Initialize Core Managers ---
            self.initialize_managers()

            # --- Initialize Background Services ---
            self.initialize_background_services()

            # --- Initialize Graph Tabs ---
            self.initialize_graph_tabs()


            # --- Auto-load clients ---
            self.auto_load_clients()

            # --- Initialize Strategies ---
            self.initialize_strategies()

            # --- Setup UI Connections ---
            self.setup_ui_connections()            

            # --- Auto-close at 15:31 IST ---
            self.setup_auto_close()

            # --- Strategy state recovery ---
            self.schedule_strategy_recovery()

            logger.info("=== TRADING APPLICATION INITIALIZED SUCCESSFULLY ===")
            logger.info(f"UI displayed at {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}")

        except Exception as e:
            logger.critical(f"=== APPLICATION INITIALIZATION FAILED: {e} ===")
            raise

    def setup_ui_logging(self):
        """Setup UI logging handler"""
        try:
            logger.info("Setting up UI logging")
            logs_widget = self.ui.findChild(QPlainTextEdit, "LogsQPlainText")
            if logs_widget:
                ui_handler = QPlainTextEditHandler(logs_widget)
                formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
                ui_handler.setFormatter(formatter)
                ui_handler.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)
                logging.getLogger().addHandler(ui_handler)
                logger.info("UI logging initialized successfully")
            else:
                logger.warning("LogsQPlainText widget not found - UI logging disabled")
        except Exception as e:
            logger.error(f"Failed to setup UI logging: {e}")

    def initialize_managers(self):
        """Initialize all core manager modules"""
        try:
            logger.info("Initializing core managers")
            
            # Client Manager
            logger.info("Creating ClientManager instance")
            self.client_manager = ClientManager(self.ui)
            logger.info("ClientManager initialized")
            
            # Position Manager
            logger.info("Creating PositionManager instance")
            self.position_manager = PositionManager(self.ui, self.client_manager)
            logger.info("PositionManager initialized")
            
            # Option Loader
            logger.info("Creating OptionLoader instance")
            self.option_loader = OptionLoader(self.ui)
            logger.info("OptionLoader initialized")
            
            # Wire managers to UI
            logger.info("Wiring managers to UI")
            self.ui.client_manager = self.client_manager
            self.ui.position_manager = self.position_manager
            self.ui.option_loader = self.option_loader
            logger.info("Managers wired to UI successfully")
            
        except Exception as e:
            logger.error(f"Manager initialization failed: {e}")
            raise

    def initialize_background_services(self):
        """Initialize background services"""
        try:
            logger.info("Initializing background services")
            
            # API Status Monitor
            logger.info("Starting API Status Monitor")
            self.api_status = ApiStatus(self.ui)
            self.api_status.start()
            logger.info("API Status Monitor started")
            
            # Opstra Option Loader
            logger.info("Creating OpstraOptionLoader instance")
            self.opstra_loader = OpstraOptionLoader(self.ui, self.client_manager)
            self.ui.opstra_loader = self.opstra_loader
            logger.info("OpstraOptionLoader initialized")
            
        except Exception as e:
            logger.error(f"Background services initialization failed: {e}")

    def initialize_graph_tabs(self):
        """Initialize graph visualization tabs"""
        try:
            logger.info("Initializing graph tabs")
            
            # Price Chart Tab
            logger.info("Creating GraphPlotTab instance")
            self.graph_tab = GraphPlotTab(self.ui, self.client_manager)
            logger.info("GraphPlotTab initialized")
            
            # Payoff Graph Tab
            logger.info("Creating PayoffGraphTab instance")
            self.payoff_tab = PayoffGraphTab(self.ui, self.client_manager)
            logger.info("PayoffGraphTab initialized")
            
            # Schedule payoff calculation
            QTimer.singleShot(5000, self.payoff_tab.calculate_adjustment_points)
            logger.info("Scheduled payoff calculation")
            
        except Exception as e:
            logger.error(f"Graph tabs initialization failed: {e}")

    def initialize_strategies(self):
        """Initialize trading strategies with proper auto-run control"""
        try:
            logger.info("Initializing trading strategies")
            
            self.strategies = {}
            
            # Check if clients are available
            if not self.client_manager or not hasattr(self.client_manager, 'clients') or not self.client_manager.clients:
                logger.warning("No clients available during strategy initialization - will retry")
                # Schedule retry after client loading
                QTimer.singleShot(10000, self._retry_strategy_init)
                return
            
            # IBBM Strategy
            try:
                logger.info("Creating IBBMStrategy instance")
                ibbm_strategy = IBBMStrategy(self.ui, self.client_manager, self.position_manager)
                self.strategies["IBBM Intraday"] = ibbm_strategy
                
                # Stop IBBM auto-run if disabled
                if not self.enable_auto_run:
                    if hasattr(ibbm_strategy, 'strategy_timer') and ibbm_strategy.strategy_timer.isActive():
                        ibbm_strategy.strategy_timer.stop()
                        logger.info("IBBM strategy timer STOPPED")
                
                logger.info("IBBMStrategy initialized successfully")
            except Exception as e:
                logger.error(f"IBBMStrategy initialization failed: {e}")
            
            # Intraday Straddle Strategy - COMPLETE AUTO-RUN CONTROL
            try:
                logger.info("Creating IntradayStraddleStrategy instance")
                intraday_straddle_strategy = IntradayStraddleStrategy(self.ui, self.client_manager, self.position_manager)
                self.strategies["Intraday Straddle"] = intraday_straddle_strategy
                
                # COMPREHENSIVELY STOP ALL TIMERS AND ACTIVITY FOR INTRADAY STRADDLE
                if not ENABLE_INTRADAY_STRADDLE_AUTO_RUN:
                    # Stop all possible timers
                    timer_names = ['strategy_timer', 'monitor_timer', 'recovery_timer', 'quote_timer', 'position_timer']
                    for timer_name in timer_names:
                        timer = getattr(intraday_straddle_strategy, timer_name, None)
                        if timer and hasattr(timer, 'isActive') and timer.isActive():
                            timer.stop()
                            logger.info(f"Stopped {timer_name} for Intraday Straddle")
                    
                    logger.info("Intraday Straddle strategy COMPLETELY STOPPED - no auto-run activity")
                else:
                    logger.info("Intraday Straddle strategy auto-run ENABLED")
                
                logger.info("IntradayStraddleStrategy initialized successfully")
            except Exception as e:
                logger.error(f"IntradayStraddleStrategy initialization failed: {e}")
            
            # Monthly Straddle Strategy - COMPLETE AUTO-RUN CONTROL
            try:
                logger.info("Creating MonthlyStraddleStrategy instance")
                monthly_straddle_strategy = MonthlyStraddleStrategy(self.ui, self.client_manager, self.position_manager)
                self.strategies["Monthly Straddle"] = monthly_straddle_strategy
                
                # COMPREHENSIVELY STOP ALL TIMERS AND ACTIVITY FOR MONTHLY STRADDLE
                if not ENABLE_MONTHLY_STRADDLE_AUTO_RUN:
                    # Stop all possible timers
                    timer_names = ['strategy_timer', 'monitor_timer', 'recovery_timer', 'quote_timer', 'position_timer']
                    for timer_name in timer_names:
                        timer = getattr(monthly_straddle_strategy, timer_name, None)
                        if timer and hasattr(timer, 'isActive') and timer.isActive():
                            timer.stop()
                            logger.info(f"Stopped {timer_name} for Monthly Straddle")
                    
                    logger.info("Monthly Straddle strategy COMPLETELY STOPPED - no auto-run activity")
                else:
                    logger.info("Monthly Straddle strategy auto-run ENABLED")
                
                logger.info("MonthlyStraddleStrategy initialized successfully")
            except Exception as e:
                logger.error(f"MonthlyStraddleStrategy initialization failed: {e}")
            
            # Set position_manager reference for all strategies
            logger.info("Setting position manager references for strategies")
            for strategy_name, strategy_obj in self.strategies.items():
                if hasattr(strategy_obj, 'position_manager'):
                    strategy_obj.position_manager = self.position_manager
                    logger.debug(f"Position manager set for {strategy_name}")
            
            self.current_strategy = DEFAULT_STRATEGY
            self.enable_auto_run = ENABLE_AUTO_RUN
            logger.info(f"Current strategy set to: {self.current_strategy}")
            logger.info(f"IBBM auto-run enabled: {self.enable_auto_run}")
            logger.info(f"Intraday Straddle auto-run enabled: {ENABLE_INTRADAY_STRADDLE_AUTO_RUN}")
            logger.info(f"Monthly Straddle auto-run enabled: {ENABLE_MONTHLY_STRADDLE_AUTO_RUN}")
            
        except Exception as e:
            logger.error(f"Strategy initialization failed: {e}")

    def setup_ui_connections(self):
        """Setup UI signal connections"""
        try:
            logger.info("Setting up UI signal connections")
            
            # Add strategy options to combobox
            if hasattr(self.ui, "StrategyNameQComboBox"):
                self.ui.StrategyNameQComboBox.clear()
                self.ui.StrategyNameQComboBox.addItems(["IBBM Intraday", "Intraday Straddle", "Monthly Straddle"])
                logger.info("Strategy combobox populated with all strategies")
            
            # Opstra Loader Button
            if hasattr(self.ui, "LoadOpstraPushButton"):
                self.ui.LoadOpstraPushButton.clicked.connect(self.opstra_loader.load_opstra_data)
                logger.info("Connected LoadOpstraPushButton")
            
            # Strategy Execution
            if hasattr(self.ui, "ExecuteStrategyQPushButton") and hasattr(self.ui, "StrategyNameQComboBox"):
                self.ui.ExecuteStrategyQPushButton.clicked.connect(self.execute_selected_strategy)
                self.ui.StrategyNameQComboBox.currentTextChanged.connect(self.on_strategy_changed)
                self.ui.CurrentStrategyQLabel.setText(f"Current: {self.current_strategy}")
                logger.info("Connected strategy execution controls")
            
            # Auto-run IBBM if enabled
            if self.enable_auto_run and "IBBM Intraday" in self.strategies:
                logger.info("IBBM auto-run enabled - strategy handles its own execution")
                # Strategy handles its own timer-based execution
                
            logger.info("UI signal connections setup completed")
            
        except Exception as e:
            logger.error(f"UI connections setup failed: {e}")

    def _retry_strategy_init(self):
        """Retry strategy initialization if clients weren't ready"""
        try:
            if not self.client_manager or not hasattr(self.client_manager, 'clients') or not self.client_manager.clients:
                logger.warning("Retry strategy init - still no clients, trying again in 10s")
                QTimer.singleShot(10000, self._retry_strategy_init)
                return
                
            logger.info("Clients now available - initializing strategies")
            self.initialize_strategies()
            
        except Exception as e:
            logger.error(f"Retry strategy initialization failed: {e}")

    def auto_load_clients(self):
        """Auto-load clients at startup"""
        try:
            logger.info("Attempting auto-load of clients")
            if hasattr(self.ui, "LoadClients") and hasattr(self.ui, "load_clients_clicked"):
                self.ui.LoadClients.clicked.connect(self.ui.load_clients_clicked)
                # Trigger auto-load
                self.ui.load_clients_clicked()
                logger.info("Clients auto-loaded successfully")
            else:
                logger.warning("Client auto-load not available - manual load required")
        except Exception as e:
            logger.error(f"Client auto-load failed: {e}")

    def setup_auto_close(self):
        """Setup automatic application closure at 15:31 IST"""
        try:
            now_ist = datetime.now(IST)
            target_time = now_ist.replace(hour=15, minute=31, second=0, microsecond=0)
            if now_ist >= target_time:
                target_time += timedelta(days=1)
            delay_ms = int((target_time - now_ist).total_seconds() * 1000)
            QTimer.singleShot(delay_ms, self.cleanup)
            logger.info(f"Application will auto-close at {target_time.strftime('%Y-%m-%d %H:%M:%S IST')}")
        except Exception as e:
            logger.error(f"Auto-close setup failed: {e}")

    def on_strategy_changed(self, strategy_name):
        """Handle strategy selection change"""
        try:
            logger.info(f"Strategy changed to: {strategy_name}")
            self.current_strategy = strategy_name
            self.ui.CurrentStrategyQLabel.setText(f"Current: {strategy_name}")
            logger.debug(f"UI updated with new strategy: {strategy_name}")
        except Exception as e:
            logger.error(f"Strategy change handler failed: {e}")

    def execute_selected_strategy(self):
        """Execute the currently selected strategy with proper timer management"""
        try:
            strategy_name = self.ui.StrategyNameQComboBox.currentText()
            logger.info(f"=== MANUAL STRATEGY EXECUTION: {strategy_name} ===")
            self.current_strategy = strategy_name
            
            # For strategies with auto-run disabled, ensure timers are started ONLY for manual execution
            if strategy_name == "Intraday Straddle" and not ENABLE_INTRADAY_STRADDLE_AUTO_RUN:
                strategy_obj = self.strategies.get(strategy_name)
                if strategy_obj:
                    # Start timers only for manual execution
                    if hasattr(strategy_obj, 'strategy_timer') and not strategy_obj.strategy_timer.isActive():
                        strategy_obj.strategy_timer.start()
                        logger.info(f"Started strategy timer for {strategy_name} (manual execution)")
                    
                    if hasattr(strategy_obj, 'monitor_timer') and not strategy_obj.monitor_timer.isActive():
                        strategy_obj.monitor_timer.start()
                        logger.info(f"Started monitor timer for {strategy_name} (manual execution)")
            
            elif strategy_name == "Monthly Straddle" and not ENABLE_MONTHLY_STRADDLE_AUTO_RUN:
                strategy_obj = self.strategies.get(strategy_name)
                if strategy_obj:
                    if hasattr(strategy_obj, 'strategy_timer') and not strategy_obj.strategy_timer.isActive():
                        strategy_obj.strategy_timer.start()
                        logger.info(f"Started strategy timer for {strategy_name} (manual execution)")
            
            self.start_strategy(strategy_name)
        except Exception as e:
            logger.error(f"Strategy execution failed: {e}")
            QMessageBox.critical(self.ui, "Error", f"Failed to execute strategy: {e}")

    def start_strategy(self, strategy_name):
        """Start a specific strategy - let strategy handle its own logic"""
        try:
            logger.info(f"Starting strategy: {strategy_name}")
            strategy_obj = self.strategies.get(strategy_name)
            if strategy_obj:
                if strategy_name == "IBBM Intraday" and self.enable_auto_run:
                    logger.info("IBBM strategy auto-run enabled - strategy manages its own execution")
                
                # Let each strategy handle its own execution logic
                if hasattr(strategy_obj, "on_execute_strategy_clicked"):
                    logger.info(f"Calling on_execute_strategy_clicked for {strategy_name}")
                    strategy_obj.on_execute_strategy_clicked()
                elif hasattr(strategy_obj, "execute_strategy"):
                    logger.info(f"Calling execute_strategy for {strategy_name}")
                    strategy_obj.execute_strategy()
                else:
                    logger.warning(f"No execution method found for {strategy_name}")
            else:
                logger.error(f"Strategy not found: {strategy_name}")
        except Exception as e:
            logger.error(f"Error starting strategy {strategy_name}: {e}")

    def schedule_strategy_recovery(self):
        """Schedule strategy state recovery"""
        try:
            logger.info("Scheduling strategy state recovery")
            QTimer.singleShot(15000, self.recover_strategy_state)
            logger.info("Strategy recovery scheduled for 15 seconds after startup")
        except Exception as e:
            logger.error(f"Strategy recovery scheduling failed: {e}")

    def recover_strategy_state(self):
        """Recover strategy states - respect auto-run flags to prevent unwanted activity"""
        try:
            logger.info("=== ATTEMPTING STRATEGY STATE RECOVERY ===")
            for strategy_name, strategy_obj in self.strategies.items():
                # RESPECT AUTO-RUN FLAGS - Skip recovery for strategies with auto-run disabled
                if strategy_name == "IBBM Intraday" and not self.enable_auto_run:
                    logger.info(f"Skipping recovery for {strategy_name} - auto-run disabled")
                    continue
                elif strategy_name == "Intraday Straddle" and not ENABLE_INTRADAY_STRADDLE_AUTO_RUN:
                    logger.info(f"Skipping recovery for {strategy_name} - auto-run disabled")
                    continue
                elif strategy_name == "Monthly Straddle" and not ENABLE_MONTHLY_STRADDLE_AUTO_RUN:
                    logger.info(f"Skipping recovery for {strategy_name} - auto-run disabled")
                    continue
                
                logger.info(f"Attempting recovery for: {strategy_name}")
                try:
                    # Let each strategy handle its own recovery logic
                    if hasattr(strategy_obj, 'attempt_recovery'):
                        logger.info(f"Calling attempt_recovery for {strategy_name}")
                        success = strategy_obj.attempt_recovery()
                        logger.info(f"Recovery for {strategy_name}: {'SUCCESS' if success else 'FAILED'}")
                        
                        # If recovery succeeded for Intraday Straddle or Monthly Straddle, stop their timers if auto-run is disabled
                        if success and strategy_name in ["Intraday Straddle", "Monthly Straddle"]:
                            if strategy_name == "Intraday Straddle" and not ENABLE_INTRADAY_STRADDLE_AUTO_RUN:
                                if hasattr(strategy_obj, 'strategy_timer') and strategy_obj.strategy_timer.isActive():
                                    strategy_obj.strategy_timer.stop()
                                    logger.info(f"Stopped auto-run timer for recovered {strategy_name} (auto-run disabled)")
                                
                                if hasattr(strategy_obj, 'monitor_timer') and strategy_obj.monitor_timer.isActive():
                                    strategy_obj.monitor_timer.stop()
                                    logger.info(f"Stopped monitor timer for recovered {strategy_name} (auto-run disabled)")
                            
                            elif strategy_name == "Monthly Straddle" and not ENABLE_MONTHLY_STRADDLE_AUTO_RUN:
                                if hasattr(strategy_obj, 'strategy_timer') and strategy_obj.strategy_timer.isActive():
                                    strategy_obj.strategy_timer.stop()
                                    logger.info(f"Stopped auto-run timer for recovered {strategy_name} (auto-run disabled)")
                                    
                    elif hasattr(strategy_obj, 'recover_from_state_file'):
                        logger.info(f"Calling recover_from_state_file for {strategy_name}")
                        success = strategy_obj.recover_from_state_file()
                        logger.info(f"Recovery for {strategy_name}: {'SUCCESS' if success else 'FAILED'}")
                    else:
                        logger.warning(f"No recovery method for {strategy_name}")
                except Exception as e:
                    logger.error(f"Recovery failed for {strategy_name}: {e}")
            
            logger.info("=== STRATEGY RECOVERY COMPLETED ===")
        except Exception as e:
            logger.error(f"Strategy recovery process failed: {e}")

    def cleanup(self):
        """Cleanup application resources - let each module handle its own cleanup"""
        try:
            logger.info("=== STARTING APPLICATION CLEANUP ===")

            # Stop API Status Monitor
            if hasattr(self, "api_status"):
                logger.info("Stopping API Status Monitor")
                self.api_status.stop_monitoring()

            # Let each strategy handle its own cleanup
            for strategy_name, strategy_obj in self.strategies.items():
                logger.info(f"Cleaning up strategy: {strategy_name}")
                try:
                    # Stop strategy timers - strategy handles its own timer management
                    for timer_attr in ["strategy_timer", "monitor_timer", "recovery_timer", "quote_timer"]:
                        timer = getattr(strategy_obj, timer_attr, None)
                        if timer and hasattr(timer, 'isActive') and timer.isActive():
                            timer.stop()
                            logger.debug(f"Stopped {timer_attr} for {strategy_name}")
                    
                    # Call strategy cleanup if available
                    if hasattr(strategy_obj, 'cleanup'):
                        strategy_obj.cleanup()
                        logger.debug(f"Called cleanup for {strategy_name}")
                except Exception as e:
                    logger.error(f"Cleanup failed for {strategy_name}: {e}")

            # Stop graph updates
            if hasattr(self, "graph_tab"):
                logger.info("Stopping graph updates")
                self.graph_tab.stop_updates()
                
            if hasattr(self, "payoff_tab"):
                logger.info("Stopping payoff graph updates")
                self.payoff_tab.stop_updates()

            # Stop position manager updates
            if hasattr(self, "position_manager"):
                logger.info("Stopping position manager updates")
                self.position_manager.stop_updates()

            # Logout clients
            if hasattr(self, "client_manager"):
                logger.info("Logging out clients")
                for client_name, client_id, client in getattr(self.client_manager, "clients", []):
                    try:
                        client.logout()
                        logger.debug(f"Logged out client: {client_name}")
                    except Exception as e:
                        logger.debug(f"Error logging out {client_name}: {e}")

            logger.info("=== APPLICATION CLEANUP COMPLETED ===")
            logger.info("Exiting application normally")
            self.app.quit()

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            self.app.quit()

    def run(self):
        """Run the application"""
        try:
            logger.info(f"=== APPLICATION RUNNING ===")
            logger.info(f"Start time: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}")
            result = self.app.exec_()
            logger.info("=== APPLICATION EXITED NORMALLY ===")
            sys.exit(result)
        except Exception as e:
            logger.critical(f"=== APPLICATION CRASHED: {e} ===")
            sys.exit(1)

# ===== Entry Point =====
def main():
    """Main entry point - handles top-level exceptions"""
    try:
        logger.info("=== APPLICATION STARTING ===")
        app = TradingApp()
        app.run()
    except Exception as e:
        logger.critical(f"=== FATAL ERROR IN MAIN: {e} ===")
        try:
            QMessageBox.critical(None, "Fatal Error", f"Application failed to start: {e}")
        except:
            pass  # If UI failed, just log and exit
        sys.exit(1)

if __name__ == "__main__":
    main()