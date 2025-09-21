import os
import time
import logging
from datetime import datetime, time
from PyQt5.QtCore import QTimer
from pytz import timezone
import yfinance as yf
import pandas as pd
import math
from functools import wraps
from typing import Optional, Dict, Any, Tuple
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
    
    # Option selection parameters
    STRIKE_RANGE = 1000  # +/- around ATM
    MIN_PREMIUM = 70
    MAX_PREMIUM = 100
    
    # Stop loss parameters
    INITIAL_SL_MULTIPLIER = 1.20  # 20% SL
    SL_ROUNDING_FACTOR = 20       # Round to nearest 0.05
    
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

        # WAITING | ACTIVE | STOPPED_OUT | COMPLETED
        self.state = "WAITING"
        self.positions: Dict[str, Dict[str, Any]] = {
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
            'sl': None, 'sl_hit': False, 'entry_price': 0.0
        }

    def _reset_all_positions(self):
        self.positions = {'ce': self._empty_position(), 'pe': self._empty_position()}

    # ---------------------- Strategy Execution ----------------------
    def on_execute_strategy_clicked(self):
        current_time = ISTTimeUtils.current_time()
        strategy_name = self.ui.StrategyNameQComboBox.currentText()
        
        if (strategy_name != self.STRATEGY_NAME or
            current_time < self.TRADING_START_TIME or current_time > self.TRADING_END_TIME):
            logger.warning(f"Strategy can only run between {self.TRADING_START_TIME.strftime('%H:%M')} and {self.TRADING_END_TIME.strftime('%H:%M')}")
            return

        # Check if it's a valid entry time (15/45 minute pattern)
        if current_time.minute not in self.ENTRY_MINUTES:
            logger.warning("Strategy can only be executed at XX:15 or XX:45")
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
                    if hasattr(self.ui, 'position_manager'):
                        self.ui.position_manager._current_strategy = strategy_name
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

        if hasattr(self.ui, 'position_manager'):
            self.ui.position_manager._current_strategy = strategy_name

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
                state_file = os.path.join(self.log_dir, f"{ISTTimeUtils.current_date_str()}_ibbm_strategy_state.csv")
                
                if os.path.exists(state_file):
                    try:
                        df = pd.read_csv(state_file)
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
                
                logger.info("=== Running Strategy Cycle (Restart) ===")
                self._run_strategy_cycle()

    def _monitor_all(self):
        """Monitor both SL and positions (runs every 10 seconds)"""
        if self.state != "ACTIVE":
            return
            
        # Check API connection before proceeding
        if not self._validate_api_connection():
            logger.warning("Skipping monitoring due to API connection issues")
            return
            
        self._monitor_stop_losses()
        
        # Additional check for position validation
        if not self._positions_validated and self.positions['ce']['symbol']:
            try:
                self._validate_positions(update_prices=False)
                self._positions_validated = True
            except ValueError as e:
                logger.error(f"Position validation error: {str(e)}")
                self.state = "WAITING"  # Reset state without crashing

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
            
            # Extract scalar values from the Series
            self.current_close = last_row['Close'].item() if hasattr(last_row['Close'], 'item') else float(last_row['Close'])
            self.current_ma12 = last_row[f'MA{self.MA_WINDOW}'].item() if hasattr(last_row[f'MA{self.MA_WINDOW}'], 'item') else float(last_row[f'MA{self.MA_WINDOW}'])
            
            self.current_trend = 'up' if self.current_close > self.current_ma12 else 'down'
            logger.info(f"Market data: Close={self.current_close:.2f}, MA{self.MA_WINDOW}={self.current_ma12:.2f}, Trend={self.current_trend.upper()}")
            return True
        except Exception as e:
            logger.error(f"Market data fetch failed: {str(e)}")
            return False

    # ---------------------- Entry Logic ----------------------
    def _take_positions(self) -> bool:
        """Take positions at 9:45"""
        logger.info("Starting position taking process")
        
        # SET STRATEGY NAME FIRST - BEFORE ANY ORDERS
        if hasattr(self.ui, 'position_manager'):
            self.ui.position_manager._current_strategy = self.STRATEGY_NAME
            logger.info(f"Strategy '{self.STRATEGY_NAME}' set before order placement")
        
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
                logger.error("Failed to place CE order - checking reason")
                # Check if it's a margin issue
                if "margin" in str(ce_order_success).lower() or "insufficient" in str(ce_order_success).lower():
                    logger.error("Margin issue detected - cannot proceed with strategy")
                    return False
                logger.error("CE order failed for unknown reason - aborting strategy")
                return False
                
            # Wait a moment for order to be processed
            time.sleep(2)
            
            # Place PE order
            logger.info(f"Placing SELL order for PE: {pe_symbol}")
            pe_order_success = self._place_order(pe_symbol, pe_token, 'SELL')
            if not pe_order_success:
                logger.error("Failed to place PE order - exiting CE position")
                # Check if it's a margin issue
                if "margin" in str(pe_order_success).lower() or "insufficient" in str(pe_order_success).lower():
                    logger.error("Margin issue detected - exiting CE position")
                    self._place_order(ce_symbol, ce_token, 'BUY')
                    return False
                
                # Try to exit the CE order if PE failed
                logger.info("Exiting CE order due to PE order failure")
                exit_success = self._place_order(ce_symbol, ce_token, 'BUY')
                if not exit_success:
                    logger.error("Failed to exit CE position after PE failure")
                return False

            # Wait for positions to be updated in broker system
            logger.info("Waiting for positions to be updated in broker system...")
            time.sleep(5)  # Increased wait time for position updates

            if hasattr(self.ui, 'position_manager'):
                pm = self.ui.position_manager
                current_spot = self._get_current_spot_price()
                current_time = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
                
                # Map both CE and PE positions to IBBM strategy
                key_ce = f"{ce_symbol}_{ce_token}"
                key_pe = f"{pe_symbol}_{pe_token}"
                
                pm._strategy_symbol_token_map[key_ce] = {
                    'strategy_name': self.STRATEGY_NAME,
                    'spot_price': current_spot,
                    'timestamp': current_time
                }
                pm._strategy_symbol_token_map[key_pe] = {
                    'strategy_name': self.STRATEGY_NAME,
                    'spot_price': current_spot,
                    'timestamp': current_time
                }
                pm._save_strategy_mapping()  # Save immediately to file
                logger.info(f"Strategy mapping saved for both positions")
            # ===== END ADDITION =====

            # ONLY AFTER SUCCESSFUL ORDERS set the strategy name
            if hasattr(self.ui, 'position_manager'):
                self.ui.position_manager._current_strategy = self.STRATEGY_NAME
                logger.info(f"Strategy '{self.STRATEGY_NAME}' set after successful orders")

            # Validate positions and get actual entry prices
            logger.info("Validating positions after order placement")
            validation_attempts = 0
            positions_validated = False

            while validation_attempts < 3:
                try:
                    # Check broker positions directly
                    client = self.client_manager.clients[0][2]
                    broker_positions = client.get_positions()
                    
                    if isinstance(broker_positions, dict) and 'data' in broker_positions:
                        broker_positions = broker_positions['data']
                    
                    # Check if both positions exist in broker
                    ce_found = False
                    pe_found = False
                    ce_entry_price = 0
                    pe_entry_price = 0
                    
                    for bp in broker_positions:
                        if (bp.get('tsym') == ce_symbol and 
                            float(bp.get('netqty', 0)) < 0):  # Short position
                            ce_found = True
                            ce_entry_price = float(bp.get('avgprc', 0))
                            logger.info(f"Found CE position: {ce_symbol} @ {ce_entry_price}")
                        
                        if (bp.get('tsym') == pe_symbol and 
                            float(bp.get('netqty', 0)) < 0):  # Short position
                            pe_found = True
                            pe_entry_price = float(bp.get('avgprc', 0))
                            logger.info(f"Found PE position: {pe_symbol} @ {pe_entry_price}")
                    
                    if ce_found and pe_found:
                        # Both positions found, update our records
                        self.positions['ce']['symbol'] = ce_symbol
                        self.positions['ce']['token'] = ce_token
                        self.positions['ce']['entry_price'] = ce_entry_price
                        self.positions['ce']['ltp'] = ce_entry_price
                        
                        self.positions['pe']['symbol'] = pe_symbol
                        self.positions['pe']['token'] = pe_token
                        self.positions['pe']['entry_price'] = pe_entry_price
                        self.positions['pe']['ltp'] = pe_entry_price
                        
                        positions_validated = True
                        logger.info("Both CE and PE positions validated successfully")
                        break
                        
                    elif ce_found and not pe_found:
                        logger.warning(f"Only CE position found, missing PE: {pe_symbol}")
                        # Exit the CE position since PE is missing
                        logger.info(f"Exiting CE position due to missing PE: {ce_symbol}")
                        self._place_order(ce_symbol, ce_token, 'BUY')
                        break
                        
                    elif pe_found and not ce_found:
                        logger.warning(f"Only PE position found, missing CE: {ce_symbol}")
                        # Exit the PE position since CE is missing
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
                
                # Final check - try to exit any remaining positions
                try:
                    client = self.client_manager.clients[0][2]
                    broker_positions = client.get_positions()
                    
                    if isinstance(broker_positions, dict) and 'data' in broker_positions:
                        broker_positions = broker_positions['data']
                    
                    # Check if either position still exists and exit it
                    for bp in broker_positions:
                        if (bp.get('tsym') == ce_symbol and 
                            float(bp.get('netqty', 0)) < 0):  # Short position
                            logger.info(f"Exiting remaining CE position: {ce_symbol}")
                            self._place_order(ce_symbol, ce_token, 'BUY')
                            
                        if (bp.get('tsym') == pe_symbol and 
                            float(bp.get('netqty', 0)) < 0):  # Short position
                            logger.info(f"Exiting remaining PE position: {pe_symbol}")
                            self._place_order(pe_symbol, pe_token, 'BUY')
                            
                except Exception as e:
                    logger.error(f"Error during final position cleanup: {str(e)}")
                
                # Clear strategy name since validation failed
                if hasattr(self.ui, 'position_manager'):
                    self.ui.position_manager._current_strategy = ""
                return False

            # Calculate SL prices based on actual entry prices
            ce_sl = math.ceil(self.positions['ce']['entry_price'] * self.INITIAL_SL_MULTIPLIER * self.SL_ROUNDING_FACTOR) / self.SL_ROUNDING_FACTOR
            pe_sl = math.ceil(self.positions['pe']['entry_price'] * self.INITIAL_SL_MULTIPLIER * self.SL_ROUNDING_FACTOR) / self.SL_ROUNDING_FACTOR
            
            self.positions['ce']['sl'] = ce_sl
            self.positions['pe']['sl'] = pe_sl
            
            self.state = "ACTIVE"
            self._log_state(
                status='ACTIVE',
                comments='Positions opened',
                ce_sl_price=ce_sl,
                pe_sl_price=pe_sl
            )
            
            logger.info(f"Positions opened: CE={ce_symbol} (SL={ce_sl:.2f}), PE={pe_symbol} (SL={pe_sl:.2f})")
            return True
            
        except Exception as e:
            logger.error(f"Position setup failed: {str(e)}")
            # Clear strategy name on exception
            if hasattr(self.ui, 'position_manager'):
                self.ui.position_manager._current_strategy = ""
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

    # ---------------------- Option Selection ----------------------
    @safe_log("Option selection failed")
    def _select_options(self, expiry_date) -> Tuple[Optional[tuple], Optional[tuple]]:
        expiry_str = expiry_date.strftime("%d-%b-%Y").upper()
        logger.info(f"Selecting options for expiry: {expiry_str}")
        
        try:
            if not os.path.exists("NFO_symbols.txt"):
                logger.error("NFO_symbols.txt file not found")
                return None, None
                
            df = pd.read_csv("NFO_symbols.txt")
            nifty_options = df[(df["Instrument"] == "OPTIDX") & (df["Symbol"] == "NIFTY") & (df["Expiry"].str.strip().str.upper() == expiry_str)].copy()
            
            if len(nifty_options) == 0:
                logger.error(f"No NIFTY options found for expiry {expiry_str}")
                return None, None
            
            current_strike = round(self.current_close/100)*100
            lower_strike = current_strike - self.STRIKE_RANGE
            upper_strike = current_strike + self.STRIKE_RANGE
            
            logger.info(f"Looking for options in strike range: {lower_strike} to {upper_strike}, premium range: {self.MIN_PREMIUM} to {self.MAX_PREMIUM}")
            
            filtered_options = nifty_options[(nifty_options["StrikePrice"] >= lower_strike) & (nifty_options["StrikePrice"] <= upper_strike)]
            valid_ce, valid_pe = [], []
            
            client = self.client_manager.clients[0][2]
            
            for _, row in filtered_options.iterrows():
                symbol, token, opt_type = row["TradingSymbol"], str(row["Token"]), row["OptionType"].strip().upper()
                
                # Add retry logic for quote retrieval
                ltp = 0
                for attempt in range(3):
                    try:
                        quote = client.get_quotes('NFO', token)
                        
                        if quote is None:
                            logger.warning(f"Quote is None for {symbol} (attempt {attempt+1})")
                            time.sleep(1)
                            continue
                            
                        if not isinstance(quote, dict) or quote.get('stat') != 'Ok':
                            logger.warning(f"Invalid quote for {symbol}: {quote}")
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
                
                if not self.MIN_PREMIUM <= ltp <= self.MAX_PREMIUM:
                    logger.debug(f"Skipping {symbol} - premium {ltp} outside range {self.MIN_PREMIUM}-{self.MAX_PREMIUM}")
                    continue
                    
                if opt_type == "CE":
                    valid_ce.append((symbol, token, ltp))
                    logger.debug(f"Valid CE found: {symbol} @ {ltp:.2f}")
                else:
                    valid_pe.append((symbol, token, ltp))
                    logger.debug(f"Valid PE found: {symbol} @ {ltp:.2f}")
            
            if not valid_ce:
                logger.error(f"No valid CE options found in premium range {self.MIN_PREMIUM}-{self.MAX_PREMIUM}")
            if not valid_pe:
                logger.error(f"No valid PE options found in premium range {self.MIN_PREMIUM}-{self.MAX_PREMIUM}")
                
            if not valid_ce or not valid_pe:
                return None, None
            
            # Select options with highest premium
            ce_selected = max(valid_ce, key=lambda x: x[2])
            pe_selected = max(valid_pe, key=lambda x: x[2])
                
            logger.info(f"Selected CE: {ce_selected[0]} @ {ce_selected[2]:.2f}, PE: {pe_selected[0]} @ {pe_selected[2]:.2f}")
            return ce_selected, pe_selected
            
        except Exception as e:
            logger.error(f"Option selection error: {str(e)}")
            return None, None

    # ---------------------- Order Management ----------------------
    @safe_log("Order placement failed")
    def _place_order(self, symbol: str, token: str, action: str) -> bool:
        try:
            # Convert action to Shoonya API format
            shoonya_action = 'B' if action.upper() == 'BUY' else 'S'
            
            client = self.client_manager.clients[0][2]
            logger.info(f"Placing {action} order for {symbol} with token {token}")
            order_id = client.place_order(
                buy_or_sell=shoonya_action,
                product_type='M',
                exchange='NFO',
                tradingsymbol=symbol,
                quantity=75,
                discloseqty=0,
                price_type='MKT',
                price=0.0,
                trigger_price=0,
                retention='DAY',
                remarks=f'IBBM_{action}'
            )
            if order_id:
                logger.info(f"{action} order placed for {symbol} (ID: {order_id})")
                return True
            else:
                logger.error(f"Failed to place {action} order for {symbol}")
                return False
        except Exception as e:
            logger.error(f"Order placement error for {symbol}: {str(e)}")
            return False

    # ---------------------- Position Validation ----------------------
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

    @safe_log("Price update failed")
    def _update_position_prices(self) -> bool:
        logger.info("Updating position prices")
        client = self.client_manager.clients[0][2]
        
        for opt_type in ['ce', 'pe']:
            token = self.positions[opt_type]['token']
            if not token:
                logger.warning(f"No token found for {opt_type.upper()}")
                continue
                
            # Add retry logic for quote retrieval
            ltp = 0
            for attempt in range(3):
                try:
                    quote = client.get_quotes('NFO', token)
                    
                    if quote is None:
                        logger.warning(f"[UPDATE] Quote is None for {opt_type.upper()} (attempt {attempt+1})")
                        time.sleep(1)
                        continue
                        
                    if not isinstance(quote, dict) or quote.get('stat') != 'Ok':
                        logger.warning(f"[UPDATE] Invalid quote response for token={token}, response={quote}")
                        time.sleep(1)
                        continue
                        
                    ltp_str = quote.get('lp', '0')
                    try:
                        ltp = float(ltp_str)
                        if ltp > 0:
                            break
                    except ValueError:
                        logger.warning(f"[UPDATE] LTP conversion failed for {opt_type.upper()}: {ltp_str}")
                    
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"[UPDATE] Quote retrieval error for {opt_type.upper()}: {str(e)}")
                    time.sleep(1)
                    continue
            
            if ltp <= 0:
                logger.error(f"Failed to get valid LTP for {opt_type.upper()} after 3 attempts")
                return False
                
            self.positions[opt_type]['ltp'] = ltp
            self.positions[opt_type]['entry_price'] = ltp
            logger.info(f"{opt_type.upper()} {self.positions[opt_type]['symbol']} @ {ltp:.2f}")
                
        return True

    # ---------------------- Monitoring ----------------------
    def _monitor_trend_and_positions(self):
        if self.state != "ACTIVE":
            logger.debug("Skipping trend monitoring - not in ACTIVE state")
            return
            
        if not self._get_market_data():
            logger.error("Failed to get market data for trend monitoring")
            return
            
        logger.info(f"Trend monitoring: Trend={self.current_trend.upper()}, Close={self.current_close:.2f}, MA12={self.current_ma12:.2f}")
        
        if not self._update_position_prices():
            logger.error("Failed to update position prices for trend monitoring")
            return
            
        self._monitor_stop_losses()
        self._log_state(status='ACTIVE', comments='Regular monitoring')

    @safe_log("Stop loss monitoring failed")
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

    def _check_manual_exits(self):
        """Check if positions were manually exited using client API"""
        try:
            logger.info("Checking for manual exits")
            client = self.client_manager.clients[0][2]
            positions_response = client.get_positions()
            
            if isinstance(positions_response, dict) and positions_response.get('stat') == 'Ok':
                positions_data = positions_response.get('data', [])
            elif isinstance(positions_response, list):
                positions_data = positions_response
            else:
                logger.error("Unexpected positions response format")
                return False

            # Now safe: positions_data is always a list of dicts
            nfo_positions = [p for p in positions_data if p.get('exch') == 'NFO']

            # Check if our expected positions still exist
            active_symbols = {pos.get('tsym') for pos in nfo_positions if float(pos.get('netqty', 0)) < 0}
            expected_symbols = {self.positions['ce']['symbol'], self.positions['pe']['symbol']}
            
            if not active_symbols.intersection(expected_symbols):
                self.state = "COMPLETED"
                self._log_state(status='COMPLETED', comments='Manual exit detected')
                logger.info("Manual exit detected - strategy completed")
                
        except Exception as e:
            logger.error(f"Manual exit check failed: {str(e)}")

    # ---------------------- Exit Logic ----------------------
    @safe_log("Exit failed")
    def _exit_all_positions(self, reason: str = "Manual exit"):
        logger.info(f"Exiting all positions: {reason}")
        if self.state not in ["ACTIVE", "STOPPED_OUT"]:
            logger.warning(f"Cannot exit positions in state: {self.state}")
            return
            
        client = self.client_manager.clients[0][2]
        exit_success = True
        
        for opt_type in ['ce', 'pe']:
            symbol, token = self.positions[opt_type]['symbol'], self.positions[opt_type]['token']
            if not symbol or self.positions[opt_type]['sl_hit']:
                logger.debug(f"Skipping {opt_type.upper()} exit - no symbol or SL already hit")
                continue
                
            try:
                logger.info(f"Exiting {opt_type.upper()} {symbol}")
                if self._place_order(symbol, token, 'BUY'):
                    logger.info(f"Exited {opt_type.upper()} {symbol} ({reason})")
                    self.positions[opt_type]['sl_hit'] = True
                else:
                    logger.error(f"Failed to exit {opt_type.upper()} {symbol}")
                    exit_success = False
                    
            except Exception as e:
                logger.error(f"Exit error for {opt_type.upper()}: {str(e)}")
                exit_success = False
        
        if exit_success:
            self.state = "COMPLETED"
            # Clear strategy name after successful exit
            if hasattr(self.ui, 'position_manager'):
                self.ui.position_manager._current_strategy = ""
                logger.info("Strategy name cleared after position exit")
            self._log_state(status='COMPLETED', comments=reason)
            logger.info(f"All positions exited - {reason}")
            
            # Reset positions for next cycle
            self._reset_all_positions()
        else:
            logger.error("Failed to exit all positions successfully")
            # Don't clear strategy name if exit failed - positions may still exist

    # ---------------------- State Management ----------------------
    def _recover_state_from_file(self):
        try:
            logger.info(f"Recovering state from file: {self.current_state_file}")
            df = pd.read_csv(self.current_state_file)
            if len(df) > 0:
                last_state = df.iloc[-1]
                self.state = last_state['status']
                
                if self.state == "ACTIVE":
                    logger.info("Recovered ACTIVE state - resuming monitoring")
                    self._validate_positions(update_prices=True)
        except Exception as e:
            logger.error(f"State recovery failed: {str(e)}")
            self.state = "WAITING"

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

    # ---------------------- API Validation ----------------------
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
        
    def _get_option_ltp(self, symbol):
        """Get last traded price for an option with better error handling"""
        if not symbol:
            logger.warning("Cannot get LTP - no symbol provided")
            return 0
            
        try:
            logger.debug(f"Getting LTP for {symbol}")
            client = self.client_manager.clients[0][2]
            
            # Get token from positions or symbol lookup
            token = None
            for leg in ['ce', 'pe']:
                if self.positions[leg].get('symbol') == symbol:
                    token = self.positions[leg].get('token')
                    break
            
            if not token:
                # Try to find token from NFO symbols file
                try:
                    df = pd.read_csv("NFO_symbols.txt")
                    symbol_data = df[df["TradingSymbol"] == symbol]
                    if not symbol_data.empty:
                        token = str(symbol_data.iloc[0]['Token'])
                        logger.debug(f"Found token {token} for {symbol} in NFO file")
                except:
                    logger.warning(f"Could not find token for {symbol} in NFO file")
                    pass
            
            if not token:
                logger.warning(f"No token found for {symbol}")
                return 0
                
            # Add retry logic for quote retrieval
            for attempt in range(3):
                quote = client.get_quotes('NFO', token)
                
                # Check if quote is None or invalid
                if quote is None:
                    logger.warning(f"Quote is None for {symbol} (attempt {attempt+1})")
                    time.sleep(1)  # Wait before retry
                    continue
                    
                if not isinstance(quote, dict) or quote.get('stat') != 'Ok':
                    logger.warning(f"Invalid quote format for {symbol}: {quote}")
                    time.sleep(1)
                    continue
                    
                ltp_str = quote.get('lp')
                if not ltp_str:
                    logger.warning(f"No LTP in quote for {symbol}")
                    time.sleep(1)
                    continue
                    
                try:
                    ltp = float(ltp_str)
                    if ltp > 0:
                        logger.debug(f"Got LTP for {symbol}: {ltp}")
                        return ltp
                    else:
                        logger.warning(f"Invalid LTP value {ltp} for {symbol}")
                except ValueError:
                    logger.warning(f"LTP conversion failed for {symbol}: {ltp_str}")
                
                time.sleep(1)  # Wait before next retry
            
            logger.error(f"Failed to get LTP for {symbol} after 3 attempts")
            return 0
            
        except Exception as e:
            logger.error(f"Failed to get LTP for {symbol}: {str(e)}")
            return 0

    # ---------------------- Recovery Methods (Added to match Monthly Straddle) ----------------------
    def recover_from_positions(self, positions_list):
        """
        Recover IBBM strategy from existing positions.
        Uses the original average sell price from the broker.
        """
        try:
            logger.info(f"{self.__class__.__name__} recovering from positions: {len(positions_list)} positions")
            
            # Check if we have the required short positions
            short_ce = None
            short_pe = None
            
            for pos in positions_list:
                symbol = pos.get('symbol', '')
                avg_price = pos.get('avg_price', 0)
                net_qty = pos.get('net_qty', 0)
                
                # Determine option type
                if 'CE' in symbol or 'C' in symbol or 'C' in symbol[-6:]:
                    option_type = 'CE'
                elif 'PE' in symbol or 'P' in symbol or 'P' in symbol[-6:]:
                    option_type = 'PE'
                else:
                    option_type = None
                    logger.warning(f"Could not determine option type for symbol: {symbol}")

                if net_qty < 0:  # Short positions
                    if option_type == 'CE':
                        short_ce = {
                            'symbol': symbol, 
                            'avg_price': avg_price, 
                            'net_qty': net_qty
                        }
                    elif option_type == 'PE':
                        short_pe = {
                            'symbol': symbol, 
                            'avg_price': avg_price, 
                            'net_qty': net_qty
                        }

            logger.info(f"Short CE: {short_ce}, Short PE: {short_pe}")

            # Check if we have the core short positions
            if not short_ce or not short_pe:
                logger.warning("Missing core short positions (CE and PE)")
                return False

            # Recover core short positions
            self.state = "ACTIVE"
            self.positions["ce"] = {
                'symbol': short_ce['symbol'],
                'entry_price': short_ce['avg_price'],
                'ltp': short_ce['avg_price'],  # Use entry price as initial LTP
                'sl': math.ceil(short_ce['avg_price'] * self.INITIAL_SL_MULTIPLIER * self.SL_ROUNDING_FACTOR) / self.SL_ROUNDING_FACTOR,
                'sl_hit': False
            }
            self.positions["pe"] = {
                'symbol': short_pe['symbol'], 
                'entry_price': short_pe['avg_price'],
                'ltp': short_pe['avg_price'],  # Use entry price as initial LTP
                'sl': math.ceil(short_pe['avg_price'] * self.INITIAL_SL_MULTIPLIER * self.SL_ROUNDING_FACTOR) / self.SL_ROUNDING_FACTOR,
                'sl_hit': False
            }

            logger.info(f"Strategy RECOVERED: CE={short_ce['symbol']} @ {short_ce['avg_price']:.2f}, PE={short_pe['symbol']} @ {short_pe['avg_price']:.2f}")
            logger.info(f"CE SL: {self.positions['ce']['sl']:.2f}, PE SL: {self.positions['pe']['sl']:.2f}")
            
            # Get current market data for monitoring
            self._get_market_data()
                
            self._log_state("ACTIVE", "Strategy recovered from positions")
            return True

        except Exception as e:
            logger.error(f"Error during strategy recovery: {e}", exc_info=True)
            return False

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
            
            # Recover basic strategy state
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
            for opt_type in ['ce', 'pe']:
                symbol = latest_state.get(f'{opt_type}_symbol')
                if pd.notna(symbol) and symbol:
                    self.positions[opt_type] = {
                        'symbol': symbol,
                        'entry_price': float(latest_state.get(f'{opt_type}_entry', 0.0)),
                        'ltp': float(latest_state.get(f'{opt_type}_ltp', 0.0)),
                        'sl': float(latest_state.get(f'{opt_type}_sl', 0.0)),
                        'sl_hit': False
                    }
            
            logger.info(f"Strategy recovered from state file: {self.state}")
            logger.info(f"Recovered positions: CE={self.positions['ce'].get('symbol')}, PE={self.positions['pe'].get('symbol')}")
            
            # Get current market data
            self._get_market_data()
            
            # Log the successful recovery
            self._log_state("ACTIVE", "Recovered from state file")
            
            return True
            
        except Exception as e:
            logger.error(f"State file recovery failed: {e}")
            return False

    def _find_latest_state_file(self):
        """Find the latest state file if today's file doesn't exist"""
        try:
            # First check if today's file exists
            if os.path.exists(self.current_state_file):
                return self.current_state_file
                
            # Look for any IBBM state files
            state_files = []
            for file in os.listdir(self.log_dir):
                if file.endswith('_ibbm_strategy_state.csv'):
                    state_files.append(file)
            
            if not state_files:
                logger.info("No IBBM state files found")
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

    def _ensure_state_consistency(self):
        """Ensure state is consistent with current positions"""
        try:
            # Check if we have active positions but no state file
            has_active_positions = any(self.positions.values())
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

    def cleanup(self):
        """Cleanup strategy resources"""
        logger.info("Performing cleanup for IBBMStrategy")
        try:
            if hasattr(self.strategy_timer, "isActive") and self.strategy_timer.isActive():
                self.strategy_timer.stop()
                logger.debug("Strategy timer stopped")
                
            if hasattr(self.monitor_timer, "isActive") and self.monitor_timer.isActive():
                self.monitor_timer.stop()
                logger.debug("Monitor timer stopped")
        except Exception:
            logger.exception("Error stopping strategy timers")

        # Reset positions and state
        self.state = "WAITING"
        self.positions = {'ce': self._empty_position(), 'pe': self._empty_position()}
        self.current_close = None
        self.current_ma12 = None
        self.current_trend = None

        logger.info("Cleanup complete")


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