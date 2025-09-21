# strategy_ibbm.py - Updated with comprehensive recovery functionality

import os
import logging
from datetime import datetime, time
from PyQt5.QtCore import QTimer
from pytz import timezone
import yfinance as yf
import pandas as pd
import math
from functools import wraps
from typing import Optional, Dict, Any, Tuple, List
import traceback

IST = timezone('Asia/Kolkata')

# Get logger for this module
logger = logging.getLogger(__name__)

# ---------------------- Utility Decorators ----------------------
def safe_log(context: str):
    """Decorator to log errors gracefully without crashing."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                logger.info(f"Starting {func.__name__} - {context}")
                result = func(*args, **kwargs)
                logger.info(f"Completed {func.__name__} successfully")
                return result
            except Exception as e:
                error_msg = f"{context}: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                return None
        return wrapper
    return decorator

# ---------------------- Time Utils ----------------------
class ISTTimeUtils:
    """Utility class for Indian Standard Time operations"""
    @staticmethod
    def now() -> datetime:
        return datetime.now(IST)
    
    @staticmethod
    def current_time() -> time:
        return ISTTimeUtils.now().time()
    
    @staticmethod
    def current_date_str() -> str:
        return ISTTimeUtils.now().strftime("%Y-%m-%d")

# ---------------------- Strategy ----------------------
class IBBMStrategy:
    # ===================== CONFIGURATION CONSTANTS =====================
    STRATEGY_NAME = "IBBM Intraday"
    TRADING_START_TIME = time(9, 45)
    TRADING_END_TIME = time(14, 45)
    EOD_EXIT_TIME = time(15, 15)

    # Entry time pattern (15/45 minutes with ±1 minute tolerance)
    ENTRY_MINUTES = [14, 15, 16, 44, 45, 46]
    MONITORING_MINUTES = [15, 45]

    # Monitoring time windows
    TREND_MONITORING_START = time(9, 46)  # Start monitoring 1 minute after trading starts
    TREND_MONITORING_END = time(14, 45)   # End monitoring at trading end time

    # Timer intervals
    STRATEGY_CHECK_INTERVAL = 60000  # 1 minute in milliseconds
    MONITORING_INTERVAL = 10000      # 10 seconds in milliseconds
    # ===================== END CONFIGURATION CONSTANTS =====================
    
    # Option selection parameters
    STRIKE_RANGE = 1000  # +/- around ATM
    MIN_PREMIUM = 70
    MAX_PREMIUM = 100
    
    # Stop loss parameters - UPDATED WITH TRAILING SL LOGIC
    INITIAL_SL_MULTIPLIER = 1.20  # 20% SL
    SL_ROUNDING_FACTOR = 20       # Round to nearest 0.05
    TRAILING_SL_STEPS = [0.90, 0.80, 0.70, 0.60, 0.50]  # 10%, 20%, 30%, 40%, 50% profit steps
    
    # Hedge parameters
    HEDGE_PRICE_RANGE = (5, 15)   # Price range for hedge positions
    HEDGE_MAX_SEARCH_DISTANCE = 1000  # Max points to search for hedge strikes
    
    # Market data parameters
    YFINANCE_SYMBOL = '^NSEI'
    DATA_PERIOD = '2d'
    DATA_INTERVAL = '30m'
    MA_WINDOW = 12
    # ===================== END CONFIGURATION CONSTANTS =====================
    
    def __init__(self, ui, client_manager):
        self.ui = ui
        self.client_manager = client_manager
        self.current_close: Optional[float] = None
        self.current_ma12: Optional[float] = None
        self.current_trend: Optional[str] = None

        self.log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(self.log_dir, exist_ok=True)

        self.current_state_file = os.path.join(
            self.log_dir, f"{ISTTimeUtils.current_date_str()}_ibbm_strategy_state.csv"
        )

        # Setup timers
        self.strategy_timer = QTimer()
        self.strategy_timer.timeout.connect(self._check_and_execute_strategy)
        self.strategy_timer.start(self.STRATEGY_CHECK_INTERVAL)

        self.monitor_timer = QTimer()
        self.monitor_timer.timeout.connect(self._monitor_all)
        self.monitor_timer.start(self.MONITORING_INTERVAL)

        logger.info(f"Strategy initialized - Runs at {self.TRADING_START_TIME.strftime('%H:%M')} and monitors trend/SL till {self.TRADING_END_TIME.strftime('%H:%M')} IST")

        # WAITING | VIRTUAL_ACTIVE | ACTIVE | STOPPED_OUT | COMPLETED
        self.state = "WAITING"
        self.positions: Dict[str, Dict[str, Any]] = {
            'ce': self._empty_position(),
            'pe': self._empty_position()
        }
        
        # Store hedge positions
        self.hedge_positions: Dict[str, Dict[str, Any]] = {
            'ce': self._empty_position(),
            'pe': self._empty_position()
        }

        self.ui.ExecuteStrategyQPushButton.clicked.connect(self.on_execute_strategy_clicked)
        self._first_monitoring_logged = False 
        self._positions_validated = False
        # Try to recover from state file on initialization
        self._try_recover_from_state_file()

    # ---------------------- Helpers ----------------------
    def _empty_position(self) -> Dict[str, Any]:
        return {
            'symbol': None, 'token': None, 'ltp': 0.0,
            'sl': None, 'sl_hit': False, 'entry_price': 0.0,
            'max_profit_price': 0.0, 'trailing_step': 0,
            'initial_sl': 0.0, 'current_sl': 0.0  # Added default values
        }

    def _reset_all_positions(self):
        self.positions = {'ce': self._empty_position(), 'pe': self._empty_position()}
        self.hedge_positions = {'ce': self._empty_position(), 'pe': self._empty_position()}

    # ---------------------- Recovery ----------------------
    def _try_recover_from_state_file(self):
        """Try to recover strategy state from today's state file only"""
        try:
            if not os.path.exists(self.current_state_file):
                logger.info("No state file found for today - starting fresh")
                return
                
            logger.info(f"Attempting to recover from today's state file: {self.current_state_file}")
            
            # Read CSV with proper error handling for inconsistent fields
            try:
                df = pd.read_csv(self.current_state_file, on_bad_lines='skip')
            except:
                # Fallback: read manually if pandas fails
                with open(self.current_state_file, 'r') as f:
                    lines = f.readlines()
                
                if len(lines) <= 1:  # Only header or empty
                    logger.info("State file is empty or has only header")
                    return
                    
                # Get the last valid line (skip header)
                last_line = lines[-1].strip()
                if last_line.count(',') < 15:  # Minimum expected fields
                    logger.warning("Invalid last line in state file - skipping recovery")
                    return
                    
                # Manual parsing as fallback
                fields = last_line.split(',')
                if len(fields) < 16:
                    logger.warning(f"Insufficient fields in state file: {len(fields)}")
                    return
                    
                # Create minimal dataframe from last line
                df = pd.DataFrame([fields[:16]], columns=[
                    'timestamp', 'status', 'comments', 'nifty_price', 'nifty_ma12', 'trend',
                    'ce_symbol', 'ce_entry_price', 'ce_ltp', 'ce_sl_price', 'ce_trailing_step',
                    'pe_symbol', 'pe_entry_price', 'pe_ltp', 'pe_sl_price', 'pe_trailing_step'
                ])
            
            if len(df) == 0:
                logger.info("State file is empty - starting fresh")
                return
                
            # Get the latest state
            last_row = df.iloc[-1]
            status = last_row.get('status', 'WAITING')
            
            if status in ['COMPLETED', 'STOPPED_OUT']:
                logger.info(f"Strategy already {status} for today - waiting for tomorrow")
                self.state = status
                return
                
            if status in ['VIRTUAL_ACTIVE', 'ACTIVE']:
                logger.info(f"Recovering {status} state from today's file")
                self.state = status
                
                # Recover positions
                for leg in ['ce', 'pe']:
                    symbol_col = f"{leg}_symbol"
                    entry_price_col = f"{leg}_entry_price"
                    sl_col = f"{leg}_sl_price"
                    trailing_step_col = f"{leg}_trailing_step"
                    
                    if symbol_col in last_row and pd.notna(last_row[symbol_col]):
                        self.positions[leg]['symbol'] = last_row[symbol_col]
                        self.positions[leg]['entry_price'] = last_row.get(entry_price_col, 0)
                        self.positions[leg]['current_sl'] = last_row.get(sl_col, 0)
                        self.positions[leg]['initial_sl'] = last_row.get(sl_col, 0)
                        self.positions[leg]['trailing_step'] = last_row.get(trailing_step_col, 0)
                        self.positions[leg]['sl_hit'] = False
                        
                        # Try to get token from symbol
                        token = self._get_token_from_symbol(self.positions[leg]['symbol'])
                        if token:
                            self.positions[leg]['token'] = token
                            logger.info(f"Recovered {leg.upper()} position: {self.positions[leg]['symbol']}")
                
                # Validate positions with broker
                if self.state == "ACTIVE":
                    try:
                        self._validate_positions(update_prices=True)
                        logger.info("Positions validated with broker after recovery")
                    except ValueError as e:
                        logger.warning(f"Position validation failed after recovery: {str(e)}")
                        # Reset if positions don't exist in broker
                        self._reset_all_positions()
                        self.state = "WAITING"
                        
                logger.info(f"Successfully recovered {status} state from today's file")
                
        except Exception as e:
            logger.error(f"Failed to recover from today's state file: {str(e)}")
            self.state = "WAITING"

    def _get_token_from_symbol(self, symbol: str) -> Optional[str]:
        """Get token from symbol using NFO symbols file"""
        try:
            if not os.path.exists("NFO_symbols.txt"):
                logger.error("NFO_symbols.txt file not found")
                return None
                
            df = pd.read_csv("NFO_symbols.txt")
            symbol_data = df[df['TradingSymbol'] == symbol]
            
            if not symbol_data.empty:
                token = str(symbol_data.iloc[0]['Token'])
                logger.info(f"Found token {token} for symbol {symbol}")
                return token
                
            logger.warning(f"Token not found for symbol {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting token from symbol: {str(e)}")
            return None

    def _find_latest_state_file(self):
        """Find the latest state file if today's file doesn't exist"""
        try:
            # First check if today's file exists
            if os.path.exists(self.current_state_file):
                return self.current_state_file
                
            # Look for any IBBM strategy state files
            state_files = []
            for file in os.listdir(self.log_dir):
                if file.endswith('_ibbm_strategy_state.csv'):
                    state_files.append(file)
            
            if not state_files:
                logger.info("No IBBM strategy state files found")
                return None
                
            # Sort by date (newest first)
            state_files.sort(reverse=True)
            latest_file = state_files[0]
            latest_file_path = os.path.join(self.log_dir, latest_file)
            
            logger.info(f"Using latest state file: {latest_file}")
            return latest_file_path
            
        except Exception as e:
            logger.error(f"Error finding latest state file: {e}")
            return None

    def recover_from_state_file(self):
        """Recover strategy state from today's state CSV file only"""
        try:
            # Only use today's file
            if not os.path.exists(self.current_state_file):
                logger.info("No state file found for today - no recovery needed")
                return False
                
            # Read with error handling
            try:
                df = pd.read_csv(self.current_state_file, on_bad_lines='skip')
            except:
                logger.warning("Failed to read state file with pandas, trying manual recovery")
                return self._manual_state_file_recovery()
            
            if df.empty:
                return False
                
            latest_state = df.iloc[-1]
            if pd.isna(latest_state.get('comments')) or str(latest_state.get('comments')).strip() == "":
                logger.warning("Last row has missing comments, skipping it")
                if len(df) > 1:
                    latest_state = df.iloc[-2]
                else:
                    return False
            
            # Only recover if status was ACTIVE or VIRTUAL_ACTIVE
            if latest_state.get('status') not in ['ACTIVE', 'VIRTUAL_ACTIVE']:
                logger.info(f"Last state was {latest_state.get('status')}, not recovering")
                return False
            

            def safe_float(val, default=0.0):
                try:
                    return float(val)
                except Exception:
                    return default

            # Recover basic strategy state
            self.state = latest_state.get('status', 'WAITING')
            self.current_close = safe_float(latest_state.get('nifty_price'))
            self.current_ma12 = safe_float(latest_state.get('nifty_ma12'))
            self.current_trend = str(latest_state.get('trend', 'unknown'))
            
            # Recover positions
            for leg in ['ce', 'pe']:
                symbol_col = f"{leg}_symbol"
                entry_price_col = f"{leg}_entry_price"
                sl_col = f"{leg}_sl_price"
                trailing_step_col = f"{leg}_trailing_step"
                
                symbol = latest_state.get(symbol_col)
                if pd.notna(symbol) and symbol:
                    self.positions[leg] = {
                        'symbol': symbol,
                        'entry_price': float(latest_state.get(entry_price_col, 0.0)),
                        'current_sl': float(latest_state.get(sl_col, 0.0)),
                        'initial_sl': float(latest_state.get(sl_col, 0.0)),
                        'trailing_step': int(latest_state.get(trailing_step_col, 0)),
                        'sl_hit': False,
                        'max_profit_price': float(latest_state.get(entry_price_col, 0.0)),
                        'ltp': float(latest_state.get(entry_price_col, 0.0))
                    }
                    
                    # Try to get token from symbol
                    token = self._get_token_from_symbol(symbol)
                    if token:
                        self.positions[leg]['token'] = token
            
            logger.info(f"Strategy recovered from state file: {self.state}")
            logger.info(f"Recovered {sum(1 for p in self.positions.values() if p['symbol'])} positions")
            
            # Log the successful recovery
            self._log_state(
                self.state, 
                "Recovered from state file"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"State file recovery failed: {e}")
            return False

    def _ensure_state_consistency(self):
        """Ensure state is consistent with current positions"""
        try:
            # Check if we have active positions but no state file
            has_active_positions = any(pos['symbol'] for pos in self.positions.values())
            state_file_exists = os.path.exists(self.current_state_file)
            
            if has_active_positions and not state_file_exists:
                logger.warning("Active positions found but no state file - creating one")
                self._log_state("ACTIVE", "State file created from active positions")
                
            elif state_file_exists and not has_active_positions:
                logger.warning("State file exists but no active positions - cleaning up")
                os.remove(self.current_state_file)
                self.state = "WAITING"
                
        except Exception as e:
            logger.error(f"State consistency check failed: {e}")

    def recover_from_positions(self, positions_list: List[Dict[str, Any]]) -> bool:
        """
        Recover strategy from existing positions using the original average sell price AND entry spot price
        """
        try:
            logger.info(f"{self.__class__.__name__} recovering from positions: {len(positions_list)} positions")
            client = self._get_primary_client()
            if not client:
                return False

            sell_ce = None
            sell_pe = None
            self.total_premium_received = 0.0
            self.entry_spot_price = None  # Initialize

            for pos in positions_list:
                symbol = pos['symbol']
                avg_sell_price = pos['avg_price']
                net_qty = pos['net_qty']
                entry_spot_price = pos['entry_spot_price']  # GET THE SAVED SPOT PRICE

                # Store the first valid entry spot price we find
                if entry_spot_price and self.entry_spot_price is None:
                    self.entry_spot_price = entry_spot_price

                if 'CE' in symbol and net_qty < 0:
                    sell_ce = {'symbol': symbol, 'token': pos['token'], 'avg_price': avg_sell_price, 'net_qty': net_qty}
                    self.total_premium_received += (avg_sell_price * abs(net_qty))
                    logger.info(f"Found short CE: Sold at {avg_sell_price}, Qty: {abs(net_qty)}")
                elif 'PE' in symbol and net_qty < 0:
                    sell_pe = {'symbol': symbol, 'token': pos['token'], 'avg_price': avg_sell_price, 'net_qty': net_qty}
                    self.total_premium_received += (avg_sell_price * abs(net_qty))
                    logger.info(f"Found short PE: Sold at {avg_sell_price}, Qty: {abs(net_qty)}")

            # Check if we found the core positions
            if sell_ce and sell_pe and self.entry_spot_price:
                self.state = "ACTIVE"
                self.positions["ce"] = {
                    'symbol': sell_ce['symbol'],
                    'token': sell_ce['token'],
                    'entry_price': sell_ce['avg_price'],
                    'ltp': sell_ce['avg_price'],
                    'initial_sl': math.ceil(sell_ce['avg_price'] * self.INITIAL_SL_MULTIPLIER * self.SL_ROUNDING_FACTOR) / self.SL_ROUNDING_FACTOR,
                    'current_sl': math.ceil(sell_ce['avg_price'] * self.INITIAL_SL_MULTIPLIER * self.SL_ROUNDING_FACTOR) / self.SL_ROUNDING_FACTOR,
                    'sl_hit': False,
                    'max_profit_price': sell_ce['avg_price'],
                    'trailing_step': 0
                }
                self.positions["pe"] = {
                    'symbol': sell_pe['symbol'],
                    'token': sell_pe['token'],
                    'entry_price': sell_pe['avg_price'],
                    'ltp': sell_pe['avg_price'],
                    'initial_sl': math.ceil(sell_pe['avg_price'] * self.INITIAL_SL_MULTIPLIER * self.SL_ROUNDING_FACTOR) / self.SL_ROUNDING_FACTOR,
                    'current_sl': math.ceil(sell_pe['avg_price'] * self.INITIAL_SL_MULTIPLIER * self.SL_ROUNDING_FACTOR) / self.SL_ROUNDING_FACTOR,
                    'sl_hit': False,
                    'max_profit_price': sell_pe['avg_price'],
                    'trailing_step': 0
                }

                logger.info(f"Strategy RECOVERED. Original Premium: {self.total_premium_received:.2f}")
                logger.info(f"Original Spot: {self.entry_spot_price:.2f}")
                self._log_state("ACTIVE", "State recovered with original entry prices and spot")
                return True
            else:
                logger.warning("Recovery failed: Could not find both short CE and PE or entry spot price.")
                return False

        except Exception as e:
            logger.error(f"Error during strategy recovery: {e}", exc_info=True)
            return False

    def _get_primary_client(self):
        """Return the primary trading client"""
        try:
            if self.client_manager and hasattr(self.client_manager, "clients") and self.client_manager.clients:
                return self.client_manager.clients[0][2]
        except Exception:
            logger.exception("Error getting primary client")
        return None

    # ---------------------- Strategy Execution ----------------------
    def on_execute_strategy_clicked(self):
        current_time = ISTTimeUtils.current_time()
        strategy_name = self.ui.StrategyNameQComboBox.currentText()
        
        # Check if it's a valid entry time FIRST
        if current_time.minute not in self.ENTRY_MINUTES:
            logger.warning("Strategy can only be executed at XX:15 or XX:45")
            return

        # THEN check trading hours and strategy name
        if not (strategy_name == self.STRATEGY_NAME and
                current_time >= self.TRADING_START_TIME and 
                current_time <= self.TRADING_END_TIME):
            logger.warning("Invalid strategy name or outside trading hours")
            return

        # ONLY NOW set the strategy name (when we're actually going to execute)
        if hasattr(self.ui, 'position_manager'):
            self.ui.position_manager._current_strategy = strategy_name
            logger.info(f"Strategy '{strategy_name}' set for manual execution")

        # Check if we're already in an active state from recovery
        if self.state in ['VIRTUAL_ACTIVE', 'ACTIVE']:
            logger.info(f"Strategy already in {self.state} state - resuming monitoring")
            return

        date_str = ISTTimeUtils.current_date_str()
        positions_file = os.path.join(self.log_dir, f"{date_str}_positions.csv")

        if os.path.exists(positions_file):
            try:
                df = pd.read_csv(positions_file)
                ibbm_positions = df[
                    (df['Strategy'] == self.STRATEGY_NAME) & 
                    (df['NetQty'].astype(float) < 0)
                ]

                if not ibbm_positions.empty:
                    logger.info("Existing IBBM positions found - Starting monitoring")
                    try:
                        self._validate_positions(update_prices=True)
                        self.state = "ACTIVE"
                        logger.info("Resumed monitoring existing positions")
                        return
                    except Exception as e:
                        logger.error(f"Failed to validate existing positions: {str(e)}")
            except Exception as e:
                logger.error(f"Error reading positions file: {str(e)}")

        if self.state != "WAITING":
            logger.warning("Cannot execute strategy - not in WAITING state")
            return

        logger.info("Manually executing IBBM strategy")
        self._run_strategy_cycle()

    def _check_and_execute_strategy(self):
        current_time = ISTTimeUtils.current_time()

        # 1. End of day exit logic
        if (current_time.hour == self.EOD_EXIT_TIME.hour and 
            current_time.minute == self.EOD_EXIT_TIME.minute and 
            self.state == "ACTIVE"):
            self._exit_all_positions(reason="End of trading day")
            return

        # 2. Regular monitoring for active positions
        if ((current_time.minute in self.MONITORING_MINUTES) and 
            self.TREND_MONITORING_START <= current_time <= self.TREND_MONITORING_END and 
            self.state == "ACTIVE"):
            self._monitor_trend_and_positions()

        # 3. Strategy entry logic
        if (self.TRADING_START_TIME <= current_time <= self.TRADING_END_TIME and 
            self.state == "WAITING"):
            if current_time.minute in self.ENTRY_MINUTES:
                # Check state file for today's status
                if os.path.exists(self.current_state_file):
                    try:
                        df = pd.read_csv(self.current_state_file)
                        if len(df) > 0:
                            last_state = df.iloc[-1]
                            
                            if last_state['status'] in ['COMPLETED', 'STOPPED_OUT']:
                                logger.info("Strategy already completed for today - waiting for tomorrow")
                                return
                                
                            if last_state['status'] in ['VIRTUAL_ACTIVE', 'ACTIVE']:
                                logger.info("Resuming monitoring of existing positions after restart")
                              
                                return
                    except Exception as e:
                        logger.error(f"Error reading state file: {str(e)}")
                
                logger.info("=== Running Strategy Cycle (Restart) ===")
                self._run_strategy_cycle()

    def _monitor_all(self):
        current_time = ISTTimeUtils.current_time()
        if not (self.TRADING_START_TIME <= current_time <= self.TRADING_END_TIME):
            logger.debug("Outside trading hours, skipping monitoring")
            return
            
        if self.state not in ["VIRTUAL_ACTIVE", "ACTIVE"]:
            logger.debug(f"State is {self.state}, skipping monitoring")
            return
        
        # Check API connection before proceeding
        if not self._validate_api_connection():
            logger.warning("Skipping monitoring due to API connection issues")
            return
        
        self._check_manual_exits()
        
        if self.state == "COMPLETED":
            logger.debug("Strategy completed, skipping monitoring")
            return
        
        self._monitor_stop_losses()
        
        if self.state == "ACTIVE" and not hasattr(self, '_positions_validated') and self.positions['ce']['symbol']:
            try:
                self._validate_positions(update_prices=False)
                self._positions_validated = True
            except ValueError as e:
                logger.error(f"Position validation error: {str(e)}")
                self.state = "WAITING"

    def _run_strategy_cycle(self):
        logger.info("=== Running Strategy Cycle ===")
        
        if not self._get_market_data():
            logger.error("Failed to get market data for strategy cycle")
            return
        
        if self.state == "WAITING":
            if not self._take_positions():
                logger.error("Failed to take positions in strategy cycle")
                return
        elif self.state in ["VIRTUAL_ACTIVE", "ACTIVE"]:
            logger.info(f"Resuming strategy in {self.state} state")
        
        self._monitor_trend_and_positions()

    # ---------------------- Market Data ----------------------
    @safe_log("Data fetch failed")
    def _get_market_data(self) -> bool:
        try:
            logger.info(f"Fetching market data for {self.YFINANCE_SYMBOL}")
            data = yf.download(self.YFINANCE_SYMBOL, period=self.DATA_PERIOD, 
                          interval=self.DATA_INTERVAL, progress=False, auto_adjust=True)
            
            if len(data) == 0:
                logger.error("No data returned from yfinance")
                return False
                
            if len(data) < self.MA_WINDOW:
                logger.error(f"Insufficient data points for MA{self.MA_WINDOW} calculation. Got {len(data)}, need {self.MA_WINDOW}")
                return False
                
            data[f'MA{self.MA_WINDOW}'] = data['Close'].rolling(window=self.MA_WINDOW).mean()
            last_row = data.iloc[-1]
            self.current_close = last_row['Close'].values[0]
            self.current_ma12 = last_row[f'MA{self.MA_WINDOW}'].values[0]
            self.current_trend = 'up' if self.current_close > self.current_ma12 else 'down'
            logger.info(f"Market data: Close={self.current_close:.2f}, MA{self.MA_WINDOW}={self.current_ma12:.2f}, Trend={self.current_trend.upper()}")
            return True
        except Exception as e:
            logger.error(f"Market data fetch failed: {str(e)}")
            return False

    # ---------------------- Entry Logic ----------------------
    def _take_positions(self) -> bool:
        """Take initial positions virtually at 9:45"""
        logger.info("Starting virtual position taking process")
        
        # ===== ADD THIS CODE FIRST =====
        # SET STRATEGY NAME BEFORE ANY POSITIONS
        if hasattr(self.ui, 'position_manager'):
            self.ui.position_manager._current_strategy = self.STRATEGY_NAME
            logger.info(f"Strategy '{self.STRATEGY_NAME}' set before virtual positions")
        # ===== END ADDITION =====
        
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
            
            for leg, sym, tok, ltp in [("ce", ce_symbol, ce_token, ce_ltp), ("pe", pe_symbol, pe_token, pe_ltp)]:
                initial_sl = math.ceil(ltp * self.INITIAL_SL_MULTIPLIER * self.SL_ROUNDING_FACTOR) / self.SL_ROUNDING_FACTOR
                self.positions[leg].update({
                    'symbol': sym,
                    'token': tok,
                    'entry_price': ltp,
                    'ltp': ltp,
                    'initial_sl': initial_sl,
                    'current_sl': initial_sl,
                    'sl_hit': False,
                    'max_profit_price': ltp,
                    'trailing_step': 0
                })
                logger.info(f"Virtual {leg.upper()} position: {sym} @ {ltp:.2f}, SL={initial_sl:.2f}")
            
            self.state = "VIRTUAL_ACTIVE"
            ce_sl = self.positions['ce']['current_sl']
            pe_sl = self.positions['pe']['current_sl']
            
            self._log_state(
                status='VIRTUAL_ACTIVE',
                comments='Virtual positions opened',
                ce_sl_price=ce_sl,
                pe_sl_price=pe_sl
            )
            
            logger.info(f"Virtual positions opened: CE={ce_symbol} (SL={ce_sl:.2f}), PE={pe_symbol} (SL={pe_sl:.2f})")
            return True
            
        except Exception as e:
            logger.error(f"Virtual position setup failed: {str(e)}")
            return False

    def _find_hedge_strike_directional(self, main_strike, option_type, expiry_date_str):
        """Find hedge strike by searching directionally from main strike"""
        try:
            logger.info(f"Finding hedge strike for {option_type} from main strike {main_strike}")
            if not os.path.exists("NFO_symbols.txt"):
                logger.error("NFO_symbols.txt file not found")
                return None, None
            
            df = pd.read_csv("NFO_symbols.txt")
            
            df = df[
                (df["Instrument"].str.strip() == "OPTIDX") & 
                (df["Symbol"].str.strip() == "NIFTY") & 
                (df["Expiry"].str.strip().str.upper() == expiry_date_str)
            ]
            
            if df.empty:
                logger.error(f"No NIFTY options found for expiry {expiry_date_str}")
                return None, None
            
            all_strikes = df[df['OptionType'] == option_type]['StrikePrice'].unique()
            all_strikes = sorted(all_strikes)
            
            if option_type == 'CE':
                search_strikes = [s for s in all_strikes if s > main_strike]
                search_strikes.sort()
                search_direction = "upward"
            else:
                search_strikes = [s for s in all_strikes if s < main_strike]
                search_strikes.sort(reverse=True)
                search_direction = "downward"
            
            logger.info(f"Searching {search_direction} from {main_strike} for {option_type} hedge")
            
            search_strikes = [s for s in search_strikes if abs(s - main_strike) <= self.HEDGE_MAX_SEARCH_DISTANCE]
            
            best_strike = None
            best_ltp = float('inf')
            
            for strike in search_strikes:
                option_data = df[(df['StrikePrice'] == strike) & (df['OptionType'] == option_type)]
                if not option_data.empty:
                    symbol = option_data.iloc[0]['TradingSymbol']
                    token = option_data.iloc[0]['Token']
                    ltp = self._get_option_ltp(symbol)
                    
                    if self.HEDGE_PRICE_RANGE[0] <= ltp <= self.HEDGE_PRICE_RANGE[1]:
                        if ltp < best_ltp:
                            best_ltp = ltp
                            best_strike = strike
                            logger.debug(f"Found better {option_type} hedge strike: {strike} @ {ltp}")
                    else:
                        logger.debug(f"Strike {strike} {option_type}: {ltp} (outside range)")
            
            if best_strike:
                logger.info(f"Selected {option_type} hedge strike: {best_strike} @ {best_ltp} (lowest price)")
                return best_strike, best_ltp
            else:
                logger.warning(f"No {option_type} hedge strike found in price range {self.HEDGE_PRICE_RANGE}")
                return None, None
            
        except Exception as e:
            logger.error(f"Error finding {option_type} hedge strike: {str(e)}")
            return None, None

    def _get_option_symbol_from_chain(self, strike, option_type, expiry_date_str):
        """Get the correct trading symbol from option chain"""
        try:
            logger.info(f"Getting symbol for {option_type} strike {strike}")
            if not os.path.exists("NFO_symbols.txt"):
                logger.error("NFO_symbols.txt file not found")
                return None, None
            
            df = pd.read_csv("NFO_symbols.txt")
            
            strike_float = float(strike)
            
            option_data = df[
                (df['StrikePrice'] == strike_float) & 
                (df['OptionType'] == option_type) &
                (df["Instrument"].str.strip() == "OPTIDX") & 
                (df["Symbol"].str.strip() == "NIFTY") & 
                (df["Expiry"].str.strip().str.upper() == expiry_date_str)
            ]
            
            if not option_data.empty:
                symbol = option_data.iloc[0]['TradingSymbol']
                token = option_data.iloc[0]['Token']
                logger.info(f"Found symbol {symbol} for {option_type} strike {strike}")
                return symbol, token
            
            logger.warning(f"No symbol found for {option_type} strike {strike}")
            return None, None
            
        except Exception as e:
            logger.error(f"Error getting symbol from chain: {str(e)}")
            return None, None

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
            
            if weekday in [2, 3]:
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

    # ---------------------- Option Selection ----------------------
    @safe_log("Option selection failed")
    def _select_options(self, expiry_date) -> Tuple[Optional[tuple], Optional[tuple]]:
        expiry_str = expiry_date.strftime("%d-%b-%Y").upper()
        logger.info(f"Selecting options for expiry: {expiry_str}")
        
        try:
            df = pd.read_csv("NFO_symbols.txt")
            nifty_options = df[(df["Instrument"] == "OPTIDX") & (df["Symbol"] == "NIFTY") & (df["Expiry"].str.strip().str.upper() == expiry_str)].copy()
            
            if len(nifty_options) == 0:
                logger.error("No NIFTY options found")
                return None, None
            
            current_strike = round(self.current_close/100)*100
            lower_strike = current_strike - self.STRIKE_RANGE
            upper_strike = current_strike + self.STRIKE_RANGE
            
            logger.info(f"Looking for options in strike range: {lower_strike} to {upper_strike}, premium range: {self.MIN_PREMIUM} to {self.MAX_PREMIUM}")
            
            filtered_options = nifty_options[(nifty_options["StrikePrice"] >= lower_strike) & (nifty_options["StrikePrice"] <= upper_strike)]
            valid_ce, valid_pe = [], []
            
            for _, row in filtered_options.iterrows():
                symbol, token, opt_type = row["TradingSymbol"], str(row["Token"]), row["OptionType"].strip().upper()
                try:
                    # FIX: Use the token directly instead of relying on positions lookup
                    client = self.client_manager.clients[0][2]
                    quote = client.get_quotes('NFO', token)
                    if not quote or quote.get('stat') != 'Ok':
                        logger.warning(f"[SELECT] Invalid quote response for token={token}, response={quote}")
                        continue
                        
                    ltp = float(quote.get('lp', 0))
                    if not self.MIN_PREMIUM <= ltp <= self.MAX_PREMIUM:
                        logger.debug(f"Skipping {symbol} - premium {ltp} outside range {self.MIN_PREMIUM}-{self.MAX_PREMIUM}")
                        continue
                        
                    (valid_ce if opt_type == "CE" else valid_pe).append((symbol, token, ltp))
                    logger.debug(f"Valid {opt_type} found: {symbol} @ {ltp:.2f}")
                except Exception as e:
                    logger.warning(f"Failed to get quote for {symbol}: {str(e)}")
                    continue
            
            if not valid_ce:
                logger.error(f"No valid CE options found in premium range {self.MIN_PREMIUM}-{self.MAX_PREMIUM}")
            if not valid_pe:
                logger.error(f"No valid PE options found in premium range {self.MIN_PREMIUM}-{self.MAX_PREMIUM}")
                
            if not valid_ce or not valid_pe:
                return None, None
            
            ce_selected = max(valid_ce, key=lambda x: x[2])
            pe_selected = max(valid_pe, key=lambda x: x[2])
            logger.info(f"Selected CE: {ce_selected[0]} @ {ce_selected[2]:.2f}, PE: {pe_selected[0]} @ {pe_selected[2]:.2f}")
            return ce_selected, pe_selected
            
        except Exception as e:
            logger.error(f"Option selection failed: {str(e)}")
            return None, None

    # ---------------------- Monitoring ----------------------
    def _monitor_trend_and_positions(self):
        if not self._get_market_data():
            logger.error("Failed to get market data for trend monitoring")
            return
            
        current_time = ISTTimeUtils.current_time()
        current_trend = 'up' if self.current_close > self.current_ma12 else 'down'
        
        if ((current_time.minute in self.MONITORING_MINUTES) and 
            time(10, 0) <= current_time <= time(15, 15) and current_time.second < 10):
            self._log_state(
                status=self.state,
                comments=f"Scheduled monitoring at {current_time.strftime('%H:%M')}",
                ce_sl_price=self.positions['ce'].get('current_sl'),
                pe_sl_price=self.positions['pe'].get('current_sl'),
                trend=current_trend
            )
            
        if current_trend != self.current_trend:
            logger.info(f"Trend changed from {self.current_trend} to {current_trend}")
            self._exit_all_positions(reason="Trend reversal")
            self.state = "WAITING"
            self._log_state(status=self.state, comments=f"Trend changed to {current_trend}", trend=current_trend)
            if self.TRADING_START_TIME <= current_time <= self.TRADING_END_TIME:
                self._run_strategy_cycle()
        elif not self._first_monitoring_logged:
            self._log_state(
                status=self.state,
                comments="First monitoring check",
                ce_sl_price=self.positions['ce'].get('current_sl'),
                pe_sl_price=self.positions['pe'].get('current_sl'),
                trend=current_trend
            )
            self._first_monitoring_logged = True

    # ---------------------- SL / Exit / Entry ----------------------
    def _monitor_stop_losses(self):
        try:
            if self.state not in ["VIRTUAL_ACTIVE", "ACTIVE"]:
                logger.debug(f"State is {self.state}, skipping SL monitoring")
                return
            
            for leg in ["ce", "pe"]:
                pos = self.positions[leg]
                if not pos["symbol"]:
                    logger.debug(f"No {leg.upper()} position, skipping SL check")
                    continue
                
                ltp = self._get_option_ltp(pos["symbol"])
                if ltp is None or ltp <= 0:  # Handle None and invalid LTP
                    logger.warning(f"Invalid LTP for {leg.upper()} {pos['symbol']}, skipping SL check")
                    continue
                    
                pos["ltp"] = ltp
                
                # Update max profit price for trailing SL
                if (self.state == "ACTIVE" and leg in self.positions and 
                    pos.get('entry_price') is not None):
                    current_max = pos.get('max_profit_price', float('inf'))
                    if current_max is None:  # Handle None case
                        pos['max_profit_price'] = ltp
                        current_max = ltp
                    
                    if ltp < current_max:
                        pos['max_profit_price'] = ltp
                        self._update_trailing_sl(leg, ltp)
                
                # Virtual -> Real transition - ensure current_sl is not None
                current_sl = pos.get("current_sl")
                if current_sl is None:
                    logger.warning(f"No current SL set for {leg.upper()}, skipping check")
                    continue
                    
                if self.state == "VIRTUAL_ACTIVE" and not pos["sl_hit"] and ltp >= current_sl:
                    logger.info(f"{leg.upper()} virtual SL hit at {ltp}. Taking real opposite leg...")
                    opposite_leg = "pe" if leg == "ce" else "ce"
                    self._enter_real_leg(opposite_leg)
                    return
                
                # Active stop loss
                if (self.state == "ACTIVE" and not pos["sl_hit"] and 
                    ltp >= current_sl):
                    self._exit_position(leg, ltp)
            
            if self.state == "ACTIVE" and not (self.positions['ce']['symbol'] or self.positions['pe']['symbol']):
                self.state = "STOPPED_OUT"
                self._log_state(status=self.state, comments="All SELL positions closed from SL")
                logger.info("All positions stopped out")
                
        except Exception as e:
            logger.error(f"SL monitoring failed: {str(e)}")

    def _exit_position(self, leg: str, ltp: float):
        """Exit only SELL positions, keep hedge positions alive"""
        pos = self.positions[leg]
        if not pos['symbol']:
            logger.warning(f"No {leg.upper()} position to exit")
            return
            
        try:
            if self._place_order(pos['symbol'], pos['token'], 'BUY'):
                logger.info(f"{leg.upper()} SL Hit at {ltp:.2f}")
                self.positions[leg] = self._empty_position()
                self._log_state(
                    status="ACTIVE",
                    comments=f"{leg.upper()} SL Hit at {ltp:.2f}",
                    **{f"{leg}_monitoring": 'Stopped'}
                )
            else:
                logger.error(f"Failed to exit {leg} position")
        except Exception as e:
            logger.error(f"Error exiting {leg} position: {str(e)}")

    def _exit_all_positions(self, reason: str = ""):
        """Exit all positions including hedge positions at market end"""
        logger.info(f"Exiting all positions: {reason}")
        all_closed = True
        
        try:
            client = self.client_manager.clients[0][2]
            broker_positions = client.get_positions()
            
            # Handle different response formats
            if isinstance(broker_positions, dict) and 'data' in broker_positions:
                broker_positions_data = broker_positions.get('data', [])
            elif isinstance(broker_positions, list):
                broker_positions_data = broker_positions  # Direct list response
            else:
                broker_positions_data = []
        except Exception as e:
            logger.error(f"Failed to get broker positions: {str(e)}")
            broker_positions_data = []
        
        for leg in ['ce', 'pe']:
            pos = self.positions[leg]
            if not pos['symbol']:
                logger.debug(f"No {leg.upper()} position to exit")
                continue
                
            position_exists = False
            for bp in broker_positions_data:
                if (bp.get('tsym') == pos['symbol'] and 
                    str(bp.get('token')) == str(pos['token']) and
                    float(bp.get('netqty', 0)) != 0):
                    position_exists = True
                    break
            
            if not position_exists:
                logger.info(f"{leg.upper()} position not found in broker - already closed?")
                self.positions[leg] = self._empty_position()
                continue
                
            try:
                if self._place_order(pos['symbol'], pos['token'], 'BUY'):
                    logger.info(f"Successfully exited {leg.upper()} position")
                    self.positions[leg] = self._empty_position()
                else:
                    logger.error(f"Failed to exit {leg.upper()} position")
                    all_closed = False
            except Exception as e:
                logger.error(f"Error exiting {leg.upper()} position: {str(e)}")
                all_closed = False
        
        # Exit hedge positions
        for leg in ['ce', 'pe']:
            pos = self.hedge_positions[leg]
            if not pos['symbol']:
                continue
                
            try:
                if self._place_order(pos['symbol'], pos['token'], 'SELL'):
                    logger.info(f"Successfully exited {leg.upper()} hedge position")
                    self.hedge_positions[leg] = self._empty_position()
                else:
                    logger.error(f"Failed to exit {leg.upper()} hedge position")
                    all_closed = False
            except Exception as e:
                logger.error(f"Error exiting {leg.upper()} hedge position: {str(e)}")
                all_closed = False
        
        if all_closed:
            self.state = "COMPLETED"
            self._log_state(status=self.state, comments=f"All positions closed: {reason}")
            logger.info(f"All positions closed successfully: {reason}")
        else:
            logger.warning("Some positions may not have been closed properly")

    def _enter_real_leg(self, leg: str):
        """Enter real position for the opposite leg when virtual SL is hit"""
        try:
            pos = self.positions[leg]
            if not pos['symbol']:
                logger.error(f"No virtual position found for {leg.upper()}")
                return False
                
            # Place SELL order for the opposite leg
            if self._place_order(pos['symbol'], pos['token'], 'SELL'):
                logger.info(f"Real {leg.upper()} position entered: {pos['symbol']}")
                self.state = "ACTIVE"
                
                # Enter hedge position for the same leg
                hedge_symbol, hedge_token, hedge_ltp = self._find_hedge_option(leg, pos['symbol'])
                if hedge_symbol and hedge_token:
                    if self._place_order(hedge_symbol, hedge_token, 'BUY'):
                        self.hedge_positions[leg].update({
                            'symbol': hedge_symbol,
                            'token': hedge_token,
                            'entry_price': hedge_ltp,
                            'ltp': hedge_ltp
                        })
                        logger.info(f"Hedge {leg.upper()} position entered: {hedge_symbol} @ {hedge_ltp}")
                
                self._log_state(
                    status=self.state,
                    comments=f"Real {leg.upper()} position entered",
                    **{f"{leg}_monitoring": 'Active'}
                )
                return True
            else:
                logger.error(f"Failed to enter real {leg.upper()} position")
                return False
                
        except Exception as e:
            logger.error(f"Error entering real {leg.upper()} position: {str(e)}")
            return False

    def _find_hedge_option(self, leg: str, main_symbol: str):
        """Find hedge option for the given leg"""
        try:
            if not os.path.exists("NFO_symbols.txt"):
                logger.error("NFO_symbols.txt file not found")
                return None, None, None
                
            df = pd.read_csv("NFO_symbols.txt")
            
            # Extract strike from main symbol
            main_strike = None
            for part in main_symbol.split('-'):
                if part.isdigit():
                    main_strike = int(part)
                    break
                    
            if not main_strike:
                logger.error(f"Could not extract strike from {main_symbol}")
                return None, None, None
                
            expiry_date = self._get_current_expiry()
            if not expiry_date:
                return None, None, None
                
            expiry_str = expiry_date.strftime("%d-%b-%Y").upper()
            
            # Find hedge strike
            hedge_strike, hedge_ltp = self._find_hedge_strike_directional(
                main_strike, leg.upper(), expiry_str
            )
            
            if not hedge_strike:
                logger.error(f"Could not find hedge strike for {leg.upper()}")
                return None, None, None
                
            # Get hedge symbol
            hedge_symbol, hedge_token = self._get_option_symbol_from_chain(
                hedge_strike, leg.upper(), expiry_str
            )
            
            if not hedge_symbol:
                logger.error(f"Could not find hedge symbol for strike {hedge_strike}")
                return None, None, None
                
            return hedge_symbol, hedge_token, hedge_ltp
            
        except Exception as e:
            logger.error(f"Error finding hedge option: {str(e)}")
            return None, None, None

    def _update_trailing_sl(self, leg: str, current_ltp: float):
        """Update trailing stop loss based on profit percentage"""
        try:
            pos = self.positions[leg]
            if not pos['symbol'] or pos['entry_price'] <= 0:
                return
                
            entry_price = pos['entry_price']
            max_profit_price = pos.get('max_profit_price', entry_price)
            
            if max_profit_price is None:
                return
                
            profit_pct = (entry_price - max_profit_price) / entry_price
            
            # Find the appropriate trailing step
            new_step = 0
            for i, step_pct in enumerate(self.TRAILING_SL_STEPS):
                if profit_pct >= step_pct:
                    new_step = i + 1
            
            current_step = pos.get('trailing_step', 0)
            
            if new_step > current_step:
                # Calculate new SL based on step
                sl_multiplier = 1.0 - (0.05 * new_step)  # 5% reduction per step
                new_sl = math.ceil(entry_price * sl_multiplier * self.SL_ROUNDING_FACTOR) / self.SL_ROUNDING_FACTOR
                
                pos['current_sl'] = new_sl
                pos['trailing_step'] = new_step
                
                logger.info(f"{leg.upper()} trailing SL updated to step {new_step}: {new_sl:.2f}")
                self._log_state(
                    status=self.state,
                    comments=f"{leg.upper()} trailing SL updated to {new_sl:.2f}",
                    **{f"{leg}_sl_price": new_sl}
                )
                
        except Exception as e:
            logger.error(f"Error updating trailing SL for {leg}: {str(e)}")

    # ---------------------- Order Placement ----------------------
    def _place_order(self, symbol: str, token: str, action: str) -> bool:
        """Place order through client manager"""
        try:
            if not symbol or not token:
                logger.error("Invalid symbol or token for order placement")
                return False
                
            client = self.client_manager.clients[0][2]
            if not client:
                logger.error("No client available for order placement")
                return False
                
            # Get current LTP for better order placement
            ltp = self._get_option_ltp(symbol)
            if not ltp:
                logger.warning(f"Could not get LTP for {symbol}, using default price")
                ltp = 100  # Default fallback
            
            # Adjust price for market orders
            if action == 'BUY':
                price = ltp * 1.05  # 5% above LTP for buy
            else:
                price = ltp * 0.95  # 5% below LTP for sell
            
            price = round(price, 1)
            shoonya_action = 'B' if action.upper() == 'BUY' else 'S'
            
            client = self.client_manager.clients[0][2]
            logger.info(f"Placing {action} order for {symbol}")
            
            order_result = client.place_order(
                            buy_or_sell=shoonya_action,
                            product_type='M',
                            exchange='NFO',
                            tradingsymbol=symbol,
                            quantity=75,
                            discloseqty=0,
                            price_type='MKT',
                            price=0.0,
                            trigger_price=None,
                            retention='DAY',
                            remarks=f"IBBM_{action.upper()}_{symbol}"
                        )
            
            if order_result and order_result.get('stat') == 'Ok':
                logger.info(f"Order placed successfully: {action} {symbol} @ {price}")
                return True
            else:
                logger.error(f"Order failed: {order_result}")
                return False
                
        except Exception as e:
            logger.error(f"Order placement error for {symbol}: {str(e)}")
            return False

    # ---------------------- Utility Methods ----------------------
    def _get_option_ltp(self, symbol: str) -> Optional[float]:
        """Get LTP for an option symbol"""
        try:
            if not symbol:
                return None
                
            client = self.client_manager.clients[0][2]
            if not client:
                return None
                
            quote = client.get_quotes('NFO', symbol)
            if not quote or quote.get('stat') != 'Ok':
                return None
                
            ltp = float(quote.get('lp', 0))
            return ltp if ltp > 0 else None
            
        except Exception as e:
            logger.error(f"Error getting LTP for {symbol}: {str(e)}")
            return None

    def _validate_positions(self, update_prices: bool = False):
        """Validate positions with broker and update if needed"""
        try:
            client = self.client_manager.clients[0][2]
            broker_positions = client.get_positions()
            
            if isinstance(broker_positions, dict) and 'data' in broker_positions:
                broker_positions = broker_positions['data']
            
            for leg in ['ce', 'pe']:
                pos = self.positions[leg]
                if not pos['symbol']:
                    continue
                    
                found = False
                for bp in broker_positions:
                    if (bp.get('tsym') == pos['symbol'] and 
                        str(bp.get('token')) == str(pos['token']) and
                        float(bp.get('netqty', 0)) != 0):
                        found = True
                        if update_prices:
                            pos['entry_price'] = float(bp.get('avgprc', 0))
                            pos['ltp'] = self._get_option_ltp(pos['symbol']) or pos['ltp']
                        break
                
                if not found:
                    raise ValueError(f"{leg.upper()} position not found in broker: {pos['symbol']}")
                    
        except Exception as e:
            logger.error(f"Position validation failed: {str(e)}")
            raise

    def _check_manual_exits(self):
        """Check if positions have been manually exited"""
        try:
            client = self.client_manager.clients[0][2]
            broker_positions = client.get_positions()
            
            if isinstance(broker_positions, dict) and 'data' in broker_positions:
                broker_positions = broker_positions['data']
            
            for leg in ['ce', 'pe']:
                pos = self.positions[leg]
                if not pos['symbol']:
                    continue
                    
                found = False
                for bp in broker_positions:
                    if (bp.get('tsym') == pos['symbol'] and 
                        str(bp.get('token')) == str(pos['token']) and
                        float(bp.get('netqty', 0)) != 0):
                        found = True
                        break
                
                if not found:
                    logger.warning(f"{leg.upper()} position manually exited: {pos['symbol']}")
                    self.positions[leg] = self._empty_position()
                    
        except Exception as e:
            logger.error(f"Manual exit check failed: {str(e)}")

    def _validate_api_connection(self) -> bool:
        """Validate that API connection is working before making requests"""
        try:
            logger.debug("Validating API connection")
            client = self.client_manager.clients[0][2]
            # Simple test call to check connectivity
            test_quote = client.get_quotes('NSE', '26000')
            if test_quote is None:
                logger.error("API connection failed - no response")
                return False
            logger.debug("API connection validated successfully")
            return True
        except Exception as e:
            logger.error(f"API connection validation failed: {str(e)}")
            return False

    # ---------------------- Logging ----------------------
    def _log_state(self, status: str, comments: str = "", **kwargs):
        """Log strategy state to CSV file"""
        try:
            current_time = ISTTimeUtils.now()
            log_data = {
                'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                'status': status,
                'comments': str(comments).replace(",", ";"),
                'nifty_price': self.current_close,
                'nifty_ma12': self.current_ma12,
                'trend': self.current_trend,
                'ce_symbol': self.positions['ce'].get('symbol'),
                'ce_entry_price': self.positions['ce'].get('entry_price'),
                'ce_ltp': self.positions['ce'].get('ltp'),
                'ce_sl_price': self.positions['ce'].get('current_sl'),
                'ce_trailing_step': self.positions['ce'].get('trailing_step', 0),
                'pe_symbol': self.positions['pe'].get('symbol'),
                'pe_entry_price': self.positions['pe'].get('entry_price'),
                'pe_ltp': self.positions['pe'].get('ltp'),
                'pe_sl_price': self.positions['pe'].get('current_sl'),
                'pe_trailing_step': self.positions['pe'].get('trailing_step', 0),
                **kwargs
            }
            
            df = pd.DataFrame([log_data])
            file_exists = os.path.exists(self.current_state_file)
            
            df.to_csv(
                self.current_state_file,
                mode='a',
                header=not file_exists,
                index=False
            )
            
            logger.info(f"State logged: {status} - {comments}")
            
        except Exception as e:
            logger.error(f"Failed to log state: {str(e)}")

