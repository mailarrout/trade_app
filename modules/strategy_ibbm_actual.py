# strategy_ibbm_actual.py
import os
import logging
import math
import time
import traceback
from datetime import datetime, time as dt_time, timedelta
from functools import wraps
from typing import Optional, Dict, Any, Tuple, List
from PyQt5.QtCore import QTimer
import pandas as pd
from pytz import timezone
import yfinance as yf

# ===== CONFIGURATION / CONSTANTS =====
IST = timezone('Asia/Kolkata')
LOGGER_NAME = __name__

STRATEGY_NAME = "IBBM Intraday"

# CURRENT (comment out the original)
TRADING_START_TIME = dt_time(9, 45)
TRADING_END_TIME = dt_time(14, 45)
EOD_EXIT_TIME = dt_time(15, 15)
ENTRY_MINUTES = [14, 15, 16, 44, 45, 46]

# NEW - Set to current time ± few minutes
# TRADING_START_TIME = dt_time(10, 0)  # Change to your current hour
# TRADING_END_TIME = dt_time(23, 59)   # Extended to end of day
# EOD_EXIT_TIME = dt_time(23, 59)      # Don't auto-exit during testing
# ENTRY_MINUTES = list(range(0, 60))   # Allow entry ANY minute

ENTRY_MODE_9_45_ONLY = False  # Set to True for 9:45 AM only, False for any monitoring minute

MONITORING_MINUTES = [15, 45]

PREMIUM_RANGE = (70.0, 100.0)
MIN_PREMIUM = PREMIUM_RANGE[0]
MAX_PREMIUM = PREMIUM_RANGE[1]

INITIAL_SL_MULTIPLIER = 1.20
SL_ROUNDING_FACTOR = 20
TRAILING_SL_STEPS = [0.10, 0.20, 0.30, 0.40, 0.50]

STRATEGY_CHECK_INTERVAL = 60000
MONITORING_INTERVAL = 10000

NFO_SYMBOLS_FILE = "NFO_symbols.txt"
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOT_SIZE = 75
ORDER_PRODUCT_TYPE = 'M'
ORDER_EXCHANGE = 'NFO'

ALLOW_REENTRY_AFTER_STOP = True

# Market Data Constants
YFINANCE_SYMBOL = "^NSEI"
DATA_PERIOD = "2d"
DATA_INTERVAL = "5m"
MA_WINDOW = 12
STRIKE_RANGE = 200

logger = logging.getLogger(LOGGER_NAME)

# ===== UTILS / DECORATORS =====
def safe_log(context: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug(f"[{context}] Entering {func.__name__}")
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"[{context}] Exception in {func.__name__}: {e}\n{traceback.format_exc()}")
                return None
        return wrapper
    return decorator

class ISTTimeUtils:
    @staticmethod
    def now() -> datetime:
        return datetime.now(IST)

    @staticmethod
    def current_time() -> dt_time:
        return ISTTimeUtils.now().time()

    @staticmethod
    def current_date_str() -> str:
        return ISTTimeUtils.now().strftime("%Y-%m-%d")

def round_sl(price: float) -> float:
    return math.ceil(price * SL_ROUNDING_FACTOR) / SL_ROUNDING_FACTOR

# ===== STRATEGY CLASS =====
class IBBMStrategy:
    def __init__(self, ui: Optional[Any], client_manager: Optional[Any], position_manager: Optional[Any] = None):
        try:
            logger.info("Initializing IBBMStrategy")
            
            self.ui = ui
            self.client_manager = client_manager
            self.position_manager = position_manager

            # === SIMPLE SAFETY CHECK ===
            if not client_manager:
                logger.warning("Client manager not available - IBBM strategy will initialize but may not function properly")
            # === END SAFETY CHECK ===

            # Initialize state file path
            self.current_state_file = os.path.join(LOG_DIR, f"{ISTTimeUtils.current_date_str()}_ibbm_actual_state.csv")
            
            self.state: str = "WAITING"
            
            self.positions: Dict[str, Dict[str, Any]] = {
                'ce': self._empty_position(),
                'pe': self._empty_position()
            }
            self.hedges: Dict[str, Dict[str, Any]] = {
                'ce': self._empty_position(),
                'pe': self._empty_position()
            }

            # Market data attributes
            self.current_close = None
            self.current_ma12 = None
            self.current_trend = None
            self.previous_trend = None

            self._positions_validated = False
            self._first_monitoring_logged = False
            self._entry_attempted = False

            # Initialize timers (don't start yet)
            self.strategy_timer = QTimer()
            self.monitor_timer = QTimer()
            
            try:
                self.strategy_timer.timeout.connect(self._check_and_execute_strategy)
                self.monitor_timer.timeout.connect(self._monitor_all)
            except Exception as e:
                logger.error(f"Failed to connect timer signals: {e}")
                raise

            # Attempt recovery BEFORE starting timers
            recovery_success = self._try_recover_from_state_file()
            
            if not recovery_success or self.state == "WAITING":
                # Create initial state file for fresh start
                self._log_state("WAITING", "Fresh start - first run of the day")
                logger.info("Strategy starting in WAITING state - ready for first entry")

            # Start timers AFTER recovery
            try:
                self.strategy_timer.start(STRATEGY_CHECK_INTERVAL)
                logger.debug("Strategy timer started.")
                
                # Only start monitor timer if we have active positions
                if self.state == "ACTIVE":
                    self.monitor_timer.start(MONITORING_INTERVAL)
                    logger.debug("Monitor timer started.")
            except Exception as e:
                logger.error(f"Failed to start timers: {e}")

            # Bind UI button
            try:
                if hasattr(self.ui, 'ExecuteStrategyQPushButton'):
                    self.ui.ExecuteStrategyQPushButton.clicked.connect(self.on_execute_strategy_clicked)
            except Exception as e:
                logger.debug(f"Could not bind UI execute button: {e}")

            logger.info(f"IBBMStrategy initialized successfully; state={self.state}")

        except Exception as e:
            logger.critical(f"CRITICAL: IBBMStrategy initialization failed: {e}")
            try:
                self._log_state("ERROR", f"Initialization failed: {e}")
            except:
                pass
            raise

    def _ensure_state_file_exists(self):
        """Ensure state file exists with initial entry"""
        try:
            if not os.path.exists(self.current_state_file):
                initial_data = {
                    'timestamp': ISTTimeUtils.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'status': 'INITIALIZING',
                    'comments': 'State file created during initialization',
                    'nifty_price': 0.0,
                    'nifty_ma12': 0.0,
                    'trend': 'unknown',
                    'ce_symbol': None,
                    'ce_entry_price': 0.0,
                    'ce_ltp': 0.0,
                    'ce_sl_price': 0.0,
                    'ce_trailing_step': 0,
                    'pe_symbol': None,
                    'pe_entry_price': 0.0,
                    'pe_ltp': 0.0,
                    'pe_sl_price': 0.0,
                    'pe_trailing_step': 0
                }
                df = pd.DataFrame([initial_data])
                df.to_csv(self.current_state_file, index=False)
                logger.info(f"Created initial state file: {self.current_state_file}")
        except Exception as e:
            logger.error(f"Failed to create initial state file: {e}")

    def _empty_position(self) -> Dict[str, Any]:
        return {
            'symbol': None, 'token': None, 'ltp': 0.0,
            'entry_price': 0.0, 'initial_sl': 0.0, 'current_sl': 0.0,
            'sl_hit': False, 'max_profit_price': 0.0, 'trailing_step': 0,
            'real_entered': False, 'order_id': None
        }

    # ===== Recovery & State Management =====
    @safe_log("recovery")
    def _try_recover_from_state_file(self):
        """Recover strategy state from CSV file - FIXED FOR FRESH START"""
        try:
            # === FIRST: Check if state file even exists ===
            if not os.path.exists(self.current_state_file):
                logger.info("No state file found for today - FRESH START")
                self.state = "WAITING"
                self._reset_all_positions()
                return False  # This is NOT recovery - it's fresh start

            # === SECOND: Check if file has valid data ===
            df = pd.read_csv(self.current_state_file)
            if df.empty:
                logger.info("State file empty for today - FRESH START")
                self.state = "WAITING"
                self._reset_all_positions()
                return False

            last = df.iloc[-1]
            last_status = str(last.get('status', 'WAITING')).strip()
            logger.info(f"Found state file with status: {last_status}")

            # === THIRD: Only consider it recovery if we have ACTIVE positions ===
            if last_status == 'ACTIVE':
                # Check if we have valid position data in the file
                ce_sym = last.get('ce_symbol', None)
                pe_sym = last.get('pe_symbol', None)
                
                # Both positions should have valid symbols for true recovery
                if (ce_sym and not pd.isna(ce_sym) and str(ce_sym).strip() and
                    pe_sym and not pd.isna(pe_sym) and str(pe_sym).strip()):
                    
                    logger.info(f"Valid recovery data found: CE={ce_sym}, PE={pe_sym}")
                    
                    # === CLIENT CHECK ===
                    if not self.client_manager:
                        logger.warning("Cannot recover - client manager not available")
                        self.state = "WAITING"
                        self._reset_all_positions()
                        return False
                    
                    # Check broker positions for ACTIVE state
                    broker_positions = self._get_broker_positions()
                    found_ce = False
                    found_pe = False

                    if broker_positions:
                        try:
                            ce_sym_str = str(ce_sym).strip()
                            pe_sym_str = str(pe_sym).strip()
                            found_ce = any(bp.get('tsym', '') == ce_sym_str for bp in broker_positions)
                            found_pe = any(bp.get('tsym', '') == pe_sym_str for bp in broker_positions)
                        except Exception as e:
                            logger.error(f"Error checking broker positions: {e}")
                            found_ce = found_pe = False

                    if found_ce and found_pe:
                        # Both positions found in broker - proceed with recovery
                        logger.info("Both positions found in broker - recovering ACTIVE state")
                        
                        # Recover position data
                        self.state = last_status

                        for leg in ['ce', 'pe']:
                            try:
                                sym = last.get(f'{leg}_symbol', None)
                                if pd.notna(sym) and str(sym).strip():
                                    self.positions[leg]['symbol'] = str(sym).strip()
                                    self.positions[leg]['entry_price'] = float(last.get(f'{leg}_entry_price', 0.0))
                                    self.positions[leg]['ltp'] = float(last.get(f'{leg}_ltp', self.positions[leg]['entry_price']))
                                    self.positions[leg]['initial_sl'] = float(last.get(f'{leg}_sl_price', 0.0))
                                    self.positions[leg]['current_sl'] = float(last.get(f'{leg}_sl_price', 0.0))
                                    self.positions[leg]['real_entered'] = True
                                    self.positions[leg]['trailing_step'] = int(last.get(f'{leg}_trailing_step', 0))
                                    
                                    token = self._get_token_from_symbol(self.positions[leg]['symbol'])
                                    if token:
                                        self.positions[leg]['token'] = token
                                        
                                    logger.info(f"Recovered {leg.upper()} main: {self.positions[leg]['symbol']}")
                            except Exception as e:
                                logger.debug(f"Failed to recover main leg {leg} from file: {e}")

                        # Recover market data
                        try:
                            self.current_close = float(last.get('nifty_price', 0.0))
                            self.current_ma12 = float(last.get('nifty_ma12', 0.0))
                            self.current_trend = last.get('trend', 'unknown')
                            self.previous_trend = self.current_trend
                        except Exception as e:
                            logger.debug(f"Failed to recover market data: {e}")

                        self._log_state("RECOVERED", f"Successfully recovered ACTIVE state")
                        return True
                        
                    else:
                        # Positions not found in broker - reset to WAITING
                        logger.warning("State file claims ACTIVE but positions not found in broker - FRESH START")
                        if ALLOW_REENTRY_AFTER_STOP:
                            logger.info("ALLOW_REENTRY_AFTER_STOP=True -> resetting to WAITING")
                            self.state = "WAITING"
                            self._reset_all_positions()
                            self._log_state("WAITING", "Reset - positions not found in broker")
                            return False
                        else:
                            logger.info("ALLOW_REENTRY_AFTER_STOP=False -> staying STOPPED_OUT")
                            self.state = "STOPPED_OUT"
                            self._reset_all_positions()
                            self._log_state("STOPPED_OUT", "Positions not found - staying STOPPED_OUT")
                            return True
                            
                else:
                    logger.warning("State file claims ACTIVE but has no valid positions - FRESH START")
                    self.state = "WAITING"
                    self._reset_all_positions()
                    return False
                    
            elif last_status in ['COMPLETED', 'STOPPED_OUT']:
                logger.info(f"Previous strategy was {last_status} - waiting for manual restart or next day")
                self.state = last_status
                self._reset_all_positions()  # Clear any position data
                return True  # This is valid recovery of completed state
                
            else:
                # WAITING or any other state - treat as fresh start
                logger.info(f"State file exists but status is {last_status} - FRESH START")
                self.state = "WAITING"
                self._reset_all_positions()
                return False

        except Exception as e:
            logger.error(f"State file recovery failed: {e}")
            # On any error, default to fresh start
            self.state = "WAITING"
            self._reset_all_positions()
            return False

    def _is_symbol_in_pos(self, symbol: str, broker_pos: dict) -> bool:
        """Check if symbol exists in broker position"""
        try:
            return broker_pos.get('tsym', '') == symbol
        except Exception:
            return False

    def _reset_all_positions(self):
        self.positions = {'ce': self._empty_position(), 'pe': self._empty_position()}
        self.hedges = {'ce': self._empty_position(), 'pe': self._empty_position()}

    # ===== Core Strategy Methods =====
    def on_execute_strategy_clicked(self):
        """Manual execution via UI button"""
        try:
            now = ISTTimeUtils.current_time()
            logger.info(f"Manual execute clicked at {now}")
            
            if now.minute not in ENTRY_MINUTES:
                logger.warning("Manual execute allowed only in entry minutes.")
                return
                
            if not (TRADING_START_TIME <= now <= TRADING_END_TIME):
                logger.warning("Manual execute outside trading hours.")
                return
                
            if self.state not in ['WAITING']:
                logger.info(f"Manual execute ignored; current state={self.state}")
                return
                
            logger.info("Manual start accepted; running entry cycle.")
            self._run_entry_cycle()
            
        except Exception as e:
            logger.error(f"Manual execute failed: {e}")

    def attempt_recovery(self):
        """Public recovery method"""
        logger.info("Attempting IBBM strategy recovery")
        try:
            if self._try_recover_from_state_file():
                logger.info("IBBM strategy recovered from state file")
                return True
            else:
                logger.info("IBBM strategy recovery failed")
                return False
        except Exception as e:
            logger.error(f"Recovery attempt failed: {e}")
            return False

    def recover_from_state_file(self):
        return self._try_recover_from_state_file()

    def recover_from_positions(self, positions_list=None):
        try:
            logger.info("Attempting to recover IBBM from positions")
            
            if positions_list is None:
                positions_list = self._get_ibbm_positions_from_broker()
            
            if not positions_list:
                logger.info("No IBBM positions found for recovery")
                return False

            ce_positions = [p for p in positions_list if 'CE' in p.get('symbol', '')]
            pe_positions = [p for p in positions_list if 'PE' in p.get('symbol', '')]

            if ce_positions:
                ce_pos = ce_positions[0]
                self.positions['ce'].update({
                    'symbol': ce_pos.get('symbol'),
                    'token': ce_pos.get('token'),
                    'entry_price': ce_pos.get('avg_price', 0),
                    'ltp': ce_pos.get('ltp', 0),
                    'real_entered': True
                })

            if pe_positions:
                pe_pos = pe_positions[0]
                self.positions['pe'].update({
                    'symbol': pe_pos.get('symbol'),
                    'token': pe_pos.get('token'),
                    'entry_price': pe_pos.get('avg_price', 0),
                    'ltp': pe_pos.get('ltp', 0),
                    'real_entered': True
                })

            if ce_positions or pe_positions:
                self.state = "ACTIVE"
                self._log_state("ACTIVE", "Recovered from broker positions")
                self.register_strategy_positions()
                logger.info("IBBM strategy recovered from broker positions")
                return True

            return False

        except Exception as e:
            logger.error(f"Error recovering from positions: {e}")
            return False

    def _get_ibbm_positions_from_broker(self):
        try:
            broker_positions = self._get_broker_positions()
            ibbm_positions = []
            
            for pos in broker_positions:
                symbol = pos.get('tsym', '')
                if 'NIFTY' in symbol and ('CE' in symbol or 'PE' in symbol):
                    net_qty = int(float(pos.get('netqty', 0)))
                    if net_qty < 0:  # Only short positions
                        ibbm_positions.append({
                            'symbol': symbol,
                            'token': pos.get('token', ''),
                            'net_qty': net_qty,
                            'avg_price': float(pos.get('netupldprc', 0)),
                            'ltp': float(pos.get('lp', 0))
                        })
            
            return ibbm_positions
        except Exception as e:
            logger.error(f"Error getting IBBM positions from broker: {e}")
            return []

    def register_strategy_positions(self, position_manager=None):
        """Register positions with PositionManager"""
        try:
            target_manager = position_manager or self.position_manager
            if not target_manager:
                logger.error("No position manager available for registration")
                return False

            spot_price = self._get_current_spot_price()
            
            for leg in ['ce', 'pe']:
                pos = self.positions[leg]
                if pos['symbol'] and pos['token']:
                    key = f"{pos['symbol']}_{pos['token']}"
                    target_manager._strategy_symbol_token_map[key] = {
                        'strategy_name': STRATEGY_NAME,
                        'spot_price': spot_price,
                        'timestamp': datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
                    }
                    logger.info(f"Registered {leg.upper()} with PositionManager: {pos['symbol']}")
            
            target_manager._save_strategy_mapping()
            logger.info("Registered IBBM strategy positions with PositionManager")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register strategy positions: {str(e)}")
            return False
        
    def execute_strategy(self):
        """Public strategy execution method"""
        logger.info("Executing IBBM strategy")
        self._run_entry_cycle()

    def _run_entry_cycle(self):
        """Run the entry cycle for strategy"""
        logger.info("Running entry cycle")
        self._run_strategy_cycle()

    def debug_strategy_flow(self):
        """Debug method to log strategy state"""
        try:
            logger.info(f"=== DEBUG STRATEGY FLOW ===")
            logger.info(f"Current time: {ISTTimeUtils.now()}")
            logger.info(f"Strategy state: {self.state}")
            logger.info(f"CE Position: {self.positions['ce']}")
            logger.info(f"PE Position: {self.positions['pe']}")
            logger.info(f"Market data - Close: {self.current_close}, MA12: {self.current_ma12}, Trend: {self.current_trend}")
            logger.info(f"=== END DEBUG ===")
        except Exception as e:
            logger.error(f"Debug flow error: {e}")

    # ===== Core Strategy Logic =====
    def _check_and_execute_strategy(self):
        current_time = ISTTimeUtils.current_time()

        if current_time.minute == 45:  # Run every 45th minute for debugging
            self.debug_strategy_flow()

        # 1. End of day exit logic
        # if (current_time >= EOD_EXIT_TIME and self.state == "ACTIVE"):
        #     self._exit_all_positions(reason="End of trading day")
        #     return

        # 2. Regular monitoring for active positions - check for trend changes
        if ((current_time.minute in MONITORING_MINUTES) and 
            TRADING_START_TIME <= current_time <= TRADING_END_TIME and 
            self.state == "ACTIVE"):
            
            # Check for trend change first
            if self._check_trend_change():
                logger.info("Trend change detected - exiting all positions")
                self._exit_all_positions(reason="Trend change")
                # Reset to allow re-entry
                self.state = "WAITING"
                self._entry_attempted = False
                self._log_state("WAITING", "Trend change - ready for re-entry")
                return
            
            self._monitor_trend_and_positions()

        # 3. Strategy entry logic - Only at 9:45 AM or if trend changes
        if (TRADING_START_TIME <= current_time <= TRADING_END_TIME and 
            self.state == "WAITING"):
            
            should_enter = False
            
            if ENTRY_MODE_9_45_ONLY:
                # === MODE 1: 9:45 AM ONLY ===
                if (current_time.hour == 9 and current_time.minute == 45 and 
                    not self._entry_attempted):
                    should_enter = True
                    self._entry_attempted = True
                    logger.info("9:45 AM entry condition met")
                
                # Or if trend changes during monitoring minutes (re-entry after trend change)
                elif current_time.minute in MONITORING_MINUTES:
                    should_enter = True
                    logger.info("Monitoring minute entry (trend change re-entry)")
                    
            else:
                # === MODE 2: ANY MONITORING MINUTE ===
                if current_time.minute in MONITORING_MINUTES:
                    should_enter = True
                    logger.info(f"Monitoring minute entry triggered at {current_time}")
                    
                    # Only set _entry_attempted for 9:45 to avoid blocking other entries
                    if current_time.hour == 9 and current_time.minute == 45:
                        self._entry_attempted = True
                        logger.info("9:45 AM first entry marked")
            
            logger.info(f"Should enter: {should_enter}")
            
            if should_enter:
                if os.path.exists(self.current_state_file):
                    try:
                        df = pd.read_csv(self.current_state_file)
                        if len(df) > 0:
                            last_state = df.iloc[-1]
                            
                            if last_state['status'] in ['COMPLETED', 'STOPPED_OUT']:
                                logger.info("Strategy already completed for today - waiting for tomorrow")
                                return
                                
                            if last_state['status'] in ['ACTIVE']:
                                logger.info("Resuming monitoring of existing positions after restart")
                                self._recover_state_from_file()
                                return
                    except Exception as e:
                        logger.error(f"Error reading state file: {str(e)}")
                
                logger.info("=== Running Strategy Cycle ===")
                self._run_strategy_cycle()

    def _check_trend_change(self) -> bool:
        """Check if trend has changed for exit"""
        if not self._get_market_data():
            return False
            
        if self.previous_trend and self.current_trend != self.previous_trend:
            logger.info(f"Trend changed from {self.previous_trend} to {self.current_trend}")
            return True
            
        return False

    def _run_strategy_cycle(self):
        logger.info("=== Running Strategy Cycle ===")
        
        if not self._get_market_data():
            logger.error("Failed to get market data for strategy cycle")
            return
        
        if self.state == "WAITING":
            if not self._take_positions():
                logger.error("Failed to take positions in strategy cycle")
                return
        elif self.state == "ACTIVE":
            logger.info(f"Resuming strategy in {self.state} state")
        
        self._monitor_trend_and_positions()

    def _monitor_all(self):
        if self.state != "ACTIVE":
            return
            
        if not self._validate_api_connection():
            logger.warning("Skipping monitoring due to API connection issues")
            return
            
        self._monitor_stop_losses()
        
        if not self._positions_validated and self.positions['ce']['symbol']:
            try:
                self._validate_positions(update_prices=True)
                self._positions_validated = True
            except ValueError as e:
                logger.error(f"Position validation error: {str(e)}")
                self.state = "WAITING"

    # ===== Market Data =====
    @safe_log("Data fetch failed")
    def _get_market_data(self) -> bool:
        try:
            logger.info(f"Fetching market data for {YFINANCE_SYMBOL}")
            data = yf.download(YFINANCE_SYMBOL, period=DATA_PERIOD, 
                            interval=DATA_INTERVAL, progress=False, auto_adjust=True)
            
            if len(data) == 0:
                logger.error("No data returned from yfinance - checking internet connection")
                return False
                
            # FIX: Calculate MA with available data points
            available_points = len(data)
            if available_points < MA_WINDOW:
                logger.warning(f"Insufficient data points for MA{MA_WINDOW}. Got {available_points}, using available data")
                # Use available data points for calculation
                ma_window = available_points
            else:
                ma_window = MA_WINDOW
            
            data[f'MA{MA_WINDOW}'] = data['Close'].rolling(window=ma_window).mean()
            last_row = data.iloc[-1]
            
            # Extract scalar values from the Series
            self.current_close = float(last_row['Close'])
            self.current_ma12 = float(last_row[f'MA{MA_WINDOW}'])
            
            self.current_trend = 'up' if self.current_close > self.current_ma12 else 'down'
            logger.info(f"Market data: Close={self.current_close:.2f}, MA{MA_WINDOW}={self.current_ma12:.2f}, Trend={self.current_trend.upper()}")
            return True
        except Exception as e:
            logger.error(f"Market data fetch failed: {str(e)}")
            return False



    # ===== Entry Logic =====
    def _take_positions(self) -> bool:
        logger.info("Starting position taking process")
        
        if self.position_manager:
            self.position_manager._current_strategy = STRATEGY_NAME
            logger.info(f"Strategy '{STRATEGY_NAME}' set before order placement")
        
        expiry_date = self._get_current_expiry()
        if not expiry_date:
            logger.error("No valid expiry date found")
            return False
        
        try:
            ce_data, pe_data = self._select_options(expiry_date)
            if not ce_data or not pe_data:
                logger.error("Could not find valid options")
                return False
            
            (ce_symbol, ce_token, ce_ltp), (pe_symbol, pe_token, pe_ltp) = ce_data, pe_data
            
            # Place CE order first
            logger.info(f"Placing SELL order for CE: {ce_symbol}")
            ce_order_success = self._place_order(ce_symbol, ce_token, 'SELL')
            if not ce_order_success:
                logger.error("Failed to place CE order")
                return False
                
            time.sleep(2)
            
            # Place PE order
            logger.info(f"Placing SELL order for PE: {pe_symbol}")
            pe_order_success = self._place_order(pe_symbol, pe_token, 'SELL')
            if not pe_order_success:
                logger.error("Failed to place PE order - exiting CE position")
                logger.info(f"Exiting CE position due to PE failure: {ce_symbol}")
                self._place_order(ce_symbol, ce_token, 'BUY')
                return False

            logger.info("Waiting for positions to be updated in broker system...")
            time.sleep(5)

            # Register with position manager
            self.register_strategy_positions()

            logger.info("Validating positions after order placement")
            validation_attempts = 0
            positions_validated = False

            while validation_attempts < 3:
                try:
                    broker_positions = self._get_broker_positions()
                    
                    ce_found = False
                    pe_found = False
                    ce_entry_price = 0
                    pe_entry_price = 0
                    
                    for bp in broker_positions:
                        if (bp.get('tsym') == ce_symbol and 
                            float(bp.get('netqty', 0)) < 0):
                            ce_found = True
                            ce_entry_price = float(bp.get('netupldprc', 0))
                            logger.info(f"Found CE position: {ce_symbol} @ {ce_entry_price}")
                        
                        if (bp.get('tsym') == pe_symbol and 
                            float(bp.get('netqty', 0)) < 0):
                            pe_found = True
                            pe_entry_price = float(bp.get('netupldprc', 0))
                            logger.info(f"Found PE position: {pe_symbol} @ {pe_entry_price}")
                    
                    if ce_found and pe_found:
                        self.positions['ce']['symbol'] = ce_symbol
                        self.positions['ce']['token'] = ce_token
                        self.positions['ce']['entry_price'] = ce_entry_price
                        self.positions['ce']['ltp'] = ce_ltp
                        self.positions['ce']['real_entered'] = True
                        
                        self.positions['pe']['symbol'] = pe_symbol
                        self.positions['pe']['token'] = pe_token
                        self.positions['pe']['entry_price'] = pe_entry_price
                        self.positions['pe']['ltp'] = pe_ltp
                        self.positions['pe']['real_entered'] = True
                        
                        positions_validated = True
                        logger.info("Both CE and PE positions validated successfully")
                        break
                        
                    elif ce_found and not pe_found:
                        logger.warning(f"Only CE position found, missing PE: {pe_symbol}")
                        logger.info(f"Exiting CE position due to missing PE: {ce_symbol}")
                        self._place_order(ce_symbol, ce_token, 'BUY')
                        break
                        
                    elif pe_found and not ce_found:
                        logger.warning(f"Only PE position found, missing CE: {ce_symbol}")
                        logger.info(f"Exiting PE position due to missing CE: {pe_symbol}")
                        self._place_order(pe_symbol, pe_token, 'BUY')
                        break
                        
                    else:
                        logger.warning("Neither CE nor PE positions found in broker")
                        
                except Exception as e:
                    logger.error(f"Error during position validation: {str(e)}")
                
                validation_attempts += 1
                logger.warning(f"Position validation failed, attempt {validation_attempts}/3")
                time.sleep(3)

            if not positions_validated:
                logger.error("Position validation failed after 3 attempts")
                
                # Cleanup any remaining positions
                try:
                    broker_positions = self._get_broker_positions()
                    for bp in broker_positions:
                        if (bp.get('tsym') == ce_symbol and 
                            float(bp.get('netqty', 0)) < 0):
                            logger.info(f"Exiting remaining CE position: {ce_symbol}")
                            self._place_order(ce_symbol, ce_token, 'BUY')
                            
                        if (bp.get('tsym') == pe_symbol and 
                            float(bp.get('netqty', 0)) < 0):
                            logger.info(f"Exiting remaining PE position: {pe_symbol}")
                            self._place_order(pe_symbol, pe_token, 'BUY')
                            
                except Exception as e:
                    logger.error(f"Error during final position cleanup: {str(e)}")
                
                if self.position_manager:
                    self.position_manager._current_strategy = ""
                return False

            # Set initial stop losses (20% each side)
            ce_sl = round_sl(self.positions['ce']['entry_price'] * INITIAL_SL_MULTIPLIER)
            pe_sl = round_sl(self.positions['pe']['entry_price'] * INITIAL_SL_MULTIPLIER)
            
            self.positions['ce']['initial_sl'] = ce_sl
            self.positions['ce']['current_sl'] = ce_sl
            self.positions['ce']['max_profit_price'] = self.positions['ce']['entry_price']
            
            self.positions['pe']['initial_sl'] = pe_sl
            self.positions['pe']['current_sl'] = pe_sl
            self.positions['pe']['max_profit_price'] = self.positions['pe']['entry_price']
            
            self.state = "ACTIVE"
            self._log_state(
                status='ACTIVE',
                comments='Positions opened with 20% SL',
                nifty_price=self.current_close,
                nifty_ma12=self.current_ma12,
                trend=self.current_trend,
                ce_sl_price=ce_sl,
                pe_sl_price=pe_sl
            )
            
            logger.info(f"Positions opened: CE={ce_symbol} (Entry={ce_entry_price:.2f}, SL={ce_sl:.2f}), PE={pe_symbol} (Entry={pe_entry_price:.2f}, SL={pe_sl:.2f})")
            return True
            
        except Exception as e:
            logger.error(f"Position setup failed: {str(e)}")
            if self.position_manager:
                self.position_manager._current_strategy = ""
            return False

    def _get_current_expiry(self):
        """Get current or next week expiry based on today and dropdown list"""
        try:
            today = datetime.now().date()
            expiry_dates = []
            logger.info(f"Getting expiry dates from dropdown, today is {today}")
            
            for i in range(self.ui.ExpiryListDropDown.count()):
                expiry_str = self.ui.ExpiryListDropDown.itemText(i)
                try:
                    expiry_date = datetime.strptime(expiry_str, "%d-%b-%Y").date()
                    if expiry_date >= today:
                        expiry_dates.append(expiry_date)
                        logger.debug(f"Found valid expiry date: {expiry_date}")
                except ValueError:
                    logger.warning(f"Could not parse expiry date: {expiry_str}")
                    continue
                    
            if not expiry_dates:
                logger.error("No valid expiry dates found in dropdown")
                return None
                
            expiry_dates.sort()
            weekday = today.weekday()
            
            if weekday in [2, 3]:  # Wed, Thu
                selected_expiry = expiry_dates[0]
                logger.info(f"Wednesday/Thursday detected, selecting current week expiry: {selected_expiry}")
            else:
                if len(expiry_dates) > 1:
                    selected_expiry = expiry_dates[1]
                    logger.info(f"Other weekday detected, selecting next week expiry: {selected_expiry}")
                else:
                    logger.warning("No next week expiry available, using current week")
                    selected_expiry = expiry_dates[0]
                    
            return selected_expiry
            
        except Exception as e:
            logger.error(f"Error getting current expiry: {str(e)}")
            return None

    # ===== Option Selection =====
    @safe_log("Option selection failed")
    def _select_options(self, expiry_date) -> Tuple[Optional[tuple], Optional[tuple]]:
        # === ADD CLIENT VALIDATION ===
        client = self._get_primary_client()
        if not client:
            logger.error("No client available for option selection - cannot proceed")
            return None, None
        
        expiry_str = expiry_date.strftime("%d-%b-%Y").upper()
        logger.info(f"Selecting options for expiry: {expiry_str}")
        
        try:
            if not os.path.exists(NFO_SYMBOLS_FILE):
                logger.error("NFO_symbols.txt file not found")
                return None, None
                
            df = pd.read_csv(NFO_SYMBOLS_FILE)
            nifty_options = df[(df["Instrument"] == "OPTIDX") & (df["Symbol"] == "NIFTY") & (df["Expiry"].str.strip().str.upper() == expiry_str)].copy()
            
            if len(nifty_options) == 0:
                logger.error(f"No NIFTY options found for expiry {expiry_str}")
                return None, None
            
            current_strike = round(self.current_close/100)*100
            lower_strike = current_strike - STRIKE_RANGE
            upper_strike = current_strike + STRIKE_RANGE
            
            logger.info(f"Looking for options in strike range: {lower_strike} to {upper_strike}, premium range: {MIN_PREMIUM} to {MAX_PREMIUM}")
            
            filtered_options = nifty_options[(nifty_options["StrikePrice"] >= lower_strike) & (nifty_options["StrikePrice"] <= upper_strike)]
            valid_ce, valid_pe = [], []
            
            client = self._get_primary_client()
            
            for _, row in filtered_options.iterrows():
                symbol, token, opt_type = row["TradingSymbol"], str(row["Token"]), row["OptionType"].strip().upper()
                
                ltp = 0
                for attempt in range(3):
                    try:
                        quote = client.get_quotes('NFO', token)
                        
                        if quote is None:
                            logger.warning(f"Quote is None for {symbol} (attempt {attempt+1})")
                            time.sleep(1)
                            continue
                            
                        if not isinstance(quote, dict):
                            logger.warning(f"Quote is not dict for {symbol}: {type(quote)}")
                            time.sleep(1)
                            continue
                            
                        if quote.get('stat') != 'Ok':
                            logger.warning(f"Quote stat not Ok for {symbol}: {quote.get('stat')}")
                            time.sleep(1)
                            continue
                            
                        ltp_str = quote.get('lp', '0')
                        try:
                            ltp = float(ltp_str)
                            if ltp > 0:
                                break
                        except ValueError:
                            logger.warning(f"LTP conversion failed for {symbol}: {ltp_str}")
                        
                        time.sleep(1)
                    except Exception as e:
                        logger.warning(f"Quote retrieval error for {symbol}: {str(e)}")
                        time.sleep(1)
                        continue
                
                if not MIN_PREMIUM <= ltp <= MAX_PREMIUM:
                    logger.debug(f"Skipping {symbol} - premium {ltp} outside range {MIN_PREMIUM}-{MAX_PREMIUM}")
                    continue
                    
                if opt_type == "CE":
                    valid_ce.append((symbol, token, ltp, row["StrikePrice"]))
                    logger.debug(f"Valid CE found: {symbol} @ {ltp:.2f}")
                else:
                    valid_pe.append((symbol, token, ltp, row["StrikePrice"]))
                    logger.debug(f"Valid PE found: {symbol} @ {ltp:.2f}")
            
            if not valid_ce or not valid_pe:
                logger.error(f"No valid options found in premium range {MIN_PREMIUM}-{MAX_PREMIUM}")
                return None, None
            
            # Select ATM options (closest to current strike)
            ce_selected = min(valid_ce, key=lambda x: abs(x[3] - current_strike))
            pe_selected = min(valid_pe, key=lambda x: abs(x[3] - current_strike))
            
            logger.info(f"Selected CE: {ce_selected[0]} (Strike={ce_selected[3]}, Premium={ce_selected[2]:.2f})")
            logger.info(f"Selected PE: {pe_selected[0]} (Strike={pe_selected[3]}, Premium={pe_selected[2]:.2f})")
            
            return (ce_selected[0], ce_selected[1], ce_selected[2]), (pe_selected[0], pe_selected[1], pe_selected[2])
            
        except Exception as e:
            logger.error(f"Option selection error: {str(e)}")
            return None, None

    # ===== Order Management =====
    def _place_order(self, symbol: str, token: str, action: str) -> bool:
        try:
            logger.info(f"Placing order: {action} {symbol} (token: {token})")
            
            client = self._get_primary_client()
            if not client:
                logger.error("No client available for order placement")
                return False
            
            # Determine order type based on action
            if action.upper() == 'SELL':
                order_type = 'S'
                product_type = ORDER_PRODUCT_TYPE
            else:  # BUY
                order_type = 'B'
                product_type = ORDER_PRODUCT_TYPE
            
            # Place order
            order_result = client.place_order(
                buy_or_sell=order_type,
                product_type=product_type,
                exchange=ORDER_EXCHANGE,
                tradingsymbol=symbol,
                quantity=LOT_SIZE,
                discloseqty=0,
                price_type='MKT',
                price=0.0,
                trigger_price=None,
                retention='DAY'
            )
            
            if order_result and order_result.get('stat') == 'Ok':
                logger.info(f"Order placed successfully: {order_type} {symbol}")
                return True
            else:
                logger.error(f"Order placement failed: {order_result}")
                return False
                
        except Exception as e:
            logger.error(f"Order placement exception: {str(e)}")
            return False

    def _exit_all_positions(self, reason: str = "Unknown"):
        logger.info(f"Exiting all positions: {reason}")
        
        exit_success = True
        
        for leg in ['ce', 'pe']:
            pos = self.positions[leg]
            if pos['symbol'] and pos['real_entered']:
                logger.info(f"Exiting {leg.upper()}: {pos['symbol']}")
                if not self._place_order(pos['symbol'], pos['token'], 'BUY'):
                    exit_success = False
                    logger.error(f"Failed to exit {leg.upper()} position")
        
        if exit_success:
            self.state = "COMPLETED"
            self._log_state("COMPLETED", f"All positions exited: {reason}")
            logger.info("All positions exited successfully")
        else:
            logger.error("Some positions failed to exit")

    def _exit_single_position(self, leg: str, reason: str = "Unknown"):
        pos = self.positions[leg]
        if pos['symbol'] and pos['real_entered']:
            logger.info(f"Exiting {leg.upper()} position: {reason}")
            if self._place_order(pos['symbol'], pos['token'], 'BUY'):
                pos['sl_hit'] = True
                self._log_state("STOPPED_OUT", f"{leg.upper()} stopped out: {reason}")
                logger.info(f"{leg.upper()} position exited successfully")
                return True
            else:
                logger.error(f"Failed to exit {leg.upper()} position")
                return False
        return True

    # ===== Stop Loss Monitoring =====
    def _monitor_stop_losses(self):
        if self.state != "ACTIVE":
            return
            
        for leg in ['ce', 'pe']:
            if not self.positions[leg]['sl_hit'] and self.positions[leg]['real_entered']:
                self._update_ltp(leg)
                self._update_stop_loss(leg)
                self._check_stop_loss(leg)

    def _update_ltp(self, leg: str):
        try:
            pos = self.positions[leg]
            if not pos['symbol'] or not pos['token']:
                return
                
            client = self._get_primary_client()
            if not client:
                return
                
            quote = client.get_quotes('NFO', pos['token'])
            if quote and quote.get('stat') == 'Ok':
                ltp_str = quote.get('lp', '0')
                try:
                    ltp = float(ltp_str)
                    pos['ltp'] = ltp
                    
                    # Update max profit price for trailing SL
                    if ltp < pos['max_profit_price']:
                        pos['max_profit_price'] = ltp
                        
                except ValueError:
                    pass
                    
        except Exception as e:
            logger.debug(f"LTP update failed for {leg}: {str(e)}")

    def _update_stop_loss(self, leg: str):
        pos = self.positions[leg]
        if pos['ltp'] == 0 or pos['max_profit_price'] == 0:
            return
            
        # Calculate profit percentage
        profit_pct = (pos['entry_price'] - pos['ltp']) / pos['entry_price']
        
        # Update trailing step based on profit
        for i, threshold in enumerate(TRAILING_SL_STEPS):
            if profit_pct >= threshold and pos['trailing_step'] < i + 1:
                pos['trailing_step'] = i + 1
                new_sl = round_sl(pos['max_profit_price'] * (1 + threshold))
                if new_sl < pos['current_sl'] or pos['current_sl'] == 0:
                    pos['current_sl'] = new_sl
                    logger.info(f"{leg.upper()} trailing SL updated to step {i+1}: {new_sl:.2f}")

    def _check_stop_loss(self, leg: str):
        pos = self.positions[leg]
        if pos['ltp'] >= pos['current_sl'] and pos['current_sl'] > 0:
            logger.info(f"{leg.upper()} stop loss hit: LTP={pos['ltp']:.2f}, SL={pos['current_sl']:.2f}")
            self._exit_single_position(leg, "Stop loss triggered")

    # ===== Position Validation =====
    def _validate_positions(self, update_prices: bool = False):
        broker_positions = self._get_broker_positions()
        
        for leg in ['ce', 'pe']:
            pos = self.positions[leg]
            if not pos['symbol']:
                continue
                
            found = False
            for bp in broker_positions:
                if (bp.get('tsym') == pos['symbol'] and 
                    float(bp.get('netqty', 0)) < 0):
                    found = True
                    if update_prices:
                        pos['entry_price'] = float(bp.get('netupldprc', 0))
                        pos['ltp'] = float(bp.get('lp', 0))
                    break
                    
            if not found:
                logger.warning(f"Position not found in broker: {pos['symbol']}")
                pos['real_entered'] = False

    # ===== Broker Communication =====
    def _get_primary_client(self):
        """Get primary client - consistent with PositionManager"""
        try:
            if (self.client_manager and 
                hasattr(self.client_manager, 'clients') and 
                self.client_manager.clients and 
                len(self.client_manager.clients) > 0):
                
                # Return the client object (third element in tuple)
                client = self.client_manager.clients[0][2]
                logger.debug(f"Primary client retrieved: {self.client_manager.clients[0][0]}")
                return client
            else:
                logger.warning("No clients available in _get_primary_client")
                return None
        except Exception as e:
            logger.error(f"Error getting primary client: {str(e)}")
            return None

    def _get_broker_positions(self):
        try:
            client = self._get_primary_client()
            if not client:
                return []
                
            positions = client.get_positions()  # ✅ Correct method name
            if positions and isinstance(positions, list):
                return positions
            return []
        except Exception as e:
            logger.error(f"Error getting broker positions: {str(e)}")
            return []
        
    def _get_token_from_symbol(self, symbol: str) -> Optional[str]:
        try:
            if not os.path.exists(NFO_SYMBOLS_FILE):
                return None
                
            df = pd.read_csv(NFO_SYMBOLS_FILE)
            match = df[df["TradingSymbol"] == symbol]
            if not match.empty:
                return str(match.iloc[0]["Token"])
            return None
        except Exception as e:
            logger.error(f"Error getting token from symbol: {str(e)}")
            return None

    def _get_current_spot_price(self) -> float:
        try:
            client = self._get_primary_client()
            if not client:
                return 0.0
                
            quote = client.get_quotes('NSE', 'NIFTY 50')
            if quote and quote.get('stat') == 'Ok':
                return float(quote.get('lp', 0))
            return 0.0
        except Exception as e:
            logger.error(f"Error getting spot price: {str(e)}")
            return 0.0

    def _validate_api_connection(self) -> bool:
        try:
            client = self._get_primary_client()
            if not client:
                return False
                
            # Simple API call to test connection
            positions = client.positions()
            return positions is not None
        except Exception:
            return False

    # ===== State Management =====
    def _log_state(self, status: str, comments: str = "", nifty_price: float = 0.0, 
               nifty_ma12: float = 0.0, trend: str = "unknown", 
               ce_sl_price: float = 0.0, pe_sl_price: float = 0.0):
        """Consistent state logging - REMOVE **extra parameter"""
        try:
            current_time = ISTTimeUtils.now()
            
            # Clean comments to prevent CSV issues
            cleaned_comments = str(comments).replace(',', ';').replace('\n', ' | ').replace('"', "'")
            
            # ALWAYS use the same 16 columns
            state_data = {
                'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                'status': str(status),
                'comments': cleaned_comments,
                'nifty_price': float(nifty_price or self.current_close or 0.0),
                'nifty_ma12': float(nifty_ma12 or self.current_ma12 or 0.0),
                'trend': str(trend or self.current_trend or "unknown"),
                'ce_symbol': str(self.positions['ce']['symbol'] or ''),
                'ce_entry_price': float(self.positions['ce']['entry_price'] or 0.0),
                'ce_ltp': float(self.positions['ce']['ltp'] or 0.0),
                'ce_sl_price': float(ce_sl_price or self.positions['ce']['current_sl'] or 0.0),
                'ce_trailing_step': int(self.positions['ce']['trailing_step'] or 0),
                'pe_symbol': str(self.positions['pe']['symbol'] or ''),
                'pe_entry_price': float(self.positions['pe']['entry_price'] or 0.0),
                'pe_ltp': float(self.positions['pe']['ltp'] or 0.0),
                'pe_sl_price': float(pe_sl_price or self.positions['pe']['current_sl'] or 0.0),
                'pe_trailing_step': int(self.positions['pe']['trailing_step'] or 0)
            }
            
            df = pd.DataFrame([state_data])
            
            if not os.path.exists(self.current_state_file):
                df.to_csv(self.current_state_file, index=False)
            else:
                df.to_csv(self.current_state_file, mode='a', header=False, index=False)
                
            logger.debug(f"State logged: {status} - {cleaned_comments}")
            
        except Exception as e:
            logger.error(f"State logging failed: {str(e)}")

    def _recover_state_from_file(self):
        return self._try_recover_from_state_file()

    def _monitor_trend_and_positions(self):
        """Monitor for trend changes and position status"""
        if self.state != "ACTIVE":
            return
            
        # Check for trend change
        if self._check_trend_change():
            logger.info("Trend change detected - exiting all positions")
            self._exit_all_positions(reason="Trend change")
            self.state = "WAITING"
            self._entry_attempted = False
            self._log_state("WAITING", "Trend change - ready for re-entry")
            return
            
        # Check if both legs are stopped out
        if (self.positions['ce']['sl_hit'] and self.positions['pe']['sl_hit']):
            logger.info("Both legs stopped out - strategy completed")
            self.state = "STOPPED_OUT"
            self._log_state("STOPPED_OUT", "Both legs stopped out")
            return
            
        # Update LTPs and check stop losses
        self._monitor_stop_losses()

    def stop(self):
        """Stop the strategy and clean up"""
        try:
            logger.info("Stopping IBBM strategy")
            
            if self.strategy_timer:
                self.strategy_timer.stop()
                
            if self.monitor_timer:
                self.monitor_timer.stop()
                
            logger.info("IBBM strategy stopped")
            
        except Exception as e:
            logger.error(f"Error stopping IBBM strategy: {str(e)}")

    # ===== Factory Function =====
    def create_strategy_instance(ui, client_manager, position_manager=None):
        try:
            logger.info("Creating IBBMStrategy instance")
            strategy = IBBMStrategy(ui, client_manager, position_manager)
            logger.info("IBBMStrategy instance created successfully")
            return strategy
        except Exception as e:
            logger.critical(f"Failed to create IBBMStrategy instance: {e}")
            return None