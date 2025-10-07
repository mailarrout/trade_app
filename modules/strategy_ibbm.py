# strategy_ibbm.py
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

# ===== CONFIGURATION / CONSTANTS =====
IST = timezone('Asia/Kolkata')
LOGGER_NAME = __name__

STRATEGY_NAME = "IBBM Intraday"

# FIXED: Changed from time() to dt_time()
TRADING_START_TIME = dt_time(9, 45)
TRADING_END_TIME = dt_time(14, 45)
EOD_EXIT_TIME = dt_time(15, 15)

ENTRY_MINUTES = [14, 15, 16, 44, 45, 46]
MONITORING_MINUTES = [15, 45]

PREMIUM_RANGE = (70.0, 100.0)
MIN_PREMIUM = PREMIUM_RANGE[0]
MAX_PREMIUM = PREMIUM_RANGE[1]
HEDGE_RANGE = (5.0, 15.0)

INITIAL_SL_MULTIPLIER = 1.20
SL_ROUNDING_FACTOR = 20
TRAILING_SL_STEPS = [0.10, 0.20, 0.30, 0.40, 0.50]

HEDGE_MAX_SEARCH_DISTANCE = 1000

STRATEGY_CHECK_INTERVAL = 60000
MONITORING_INTERVAL = 10000

NFO_SYMBOLS_FILE = "NFO_symbols.txt"
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOT_SIZE = 75
ORDER_PRODUCT_TYPE = 'M'
ORDER_EXCHANGE = 'NFO'

ALLOW_REENTRY_AFTER_STOP = True

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
    def current_time() -> dt_time:  # FIXED: Changed return type to dt_time
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

            # === ADD THIS SAFETY CHECK ===
            if not client_manager:
                logger.warning("Client manager not available - IBBM strategy will initialize but may not function properly")
            # === END SAFETY CHECK ===

            # Initialize state file FIRST to ensure recovery
            self.current_state_file = os.path.join(LOG_DIR, f"{ISTTimeUtils.current_date_str()}_ibbm_strategy_state.csv")
            
            # Create initial state file entry immediately
            self._ensure_state_file_exists()
            
            self.state: str = "WAITING"

            self.positions: Dict[str, Dict[str, Any]] = {
                'ce': self._empty_position(),
                'pe': self._empty_position()
            }
            self.hedges: Dict[str, Dict[str, Any]] = {
                'ce': self._empty_position(),
                'pe': self._empty_position()
            }

            self._positions_validated = False
            self._first_monitoring_logged = False

            # Initialize timers with error handling
            self.strategy_timer = QTimer()
            self.monitor_timer = QTimer()
            
            try:
                self.strategy_timer.timeout.connect(self._check_and_execute_strategy)
                self.strategy_timer.start(STRATEGY_CHECK_INTERVAL)
                logger.debug("Strategy timer started.")
            except Exception as e:
                logger.error(f"Failed to start strategy timer: {e}")

            try:
                self.monitor_timer.timeout.connect(self._monitor_all)
                self.monitor_timer.start(MONITORING_INTERVAL)
                logger.debug("Monitor timer started.")
            except Exception as e:
                logger.error(f"Failed to start monitor timer: {e}")

            # Bind UI button if present
            try:
                if hasattr(self.ui, 'ExecuteStrategyQPushButton'):
                    self.ui.ExecuteStrategyQPushButton.clicked.connect(self.on_execute_strategy_clicked)
            except Exception as e:
                logger.debug(f"Could not bind UI execute button: {e}")

            # Attempt recovery
            recovery_success = self._try_recover_from_state_file()
            if not recovery_success:
                self._log_state("WAITING", "Fresh start - no recovery data found")
            
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
                    'ce_symbol': None,
                    'ce_entry_price': 0.0,
                    'ce_ltp': 0.0,
                    'ce_sl_price': 0.0,
                    'ce_real_entered': False,
                    'pe_symbol': None,
                    'pe_entry_price': 0.0,
                    'pe_ltp': 0.0,
                    'pe_sl_price': 0.0,
                    'pe_real_entered': False,
                    'ce_hedge_symbol': None,
                    'ce_hedge_entry': 0.0,
                    'pe_hedge_symbol': None,
                    'pe_hedge_entry': 0.0
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
            'real_entered': False
        }

    # ===== Recovery & State Management =====
    @safe_log("recovery")
    def _try_recover_from_state_file(self):
        """Recover strategy state from CSV file with enhanced error handling"""
        try:
            # === ADD CLIENT CHECK ===
            if not self.client_manager:
                logger.warning("Cannot recover - client manager not available")
                return False
            # === END CLIENT CHECK ===
            
            if not os.path.exists(self.current_state_file):
                logger.info("No state file found for today; starting fresh.")
                self._log_state("WAITING", "No state file found - fresh start")
                return False

            df = pd.read_csv(self.current_state_file)
            if df.empty:
                logger.info("State file empty for today.")
                self._log_state("WAITING", "Empty state file - fresh start")
                return False

            last = df.iloc[-1]
            last_status = str(last.get('status', 'WAITING')).strip()
            logger.info(f"Recovered last status from file: {last_status}")

            # Validate broker positions for ACTIVE/STOPPED_OUT states
            if last_status in ['ACTIVE', 'STOPPED_OUT']:
                broker_positions = self._get_broker_positions()
                ce_sym = last.get('ce_symbol', None)
                pe_sym = last.get('pe_symbol', None)
                found_ce = False
                found_pe = False

                if broker_positions:
                    try:
                        if ce_sym and not pd.isna(ce_sym):
                            found_ce = any(self._is_symbol_in_pos(ce_sym, bp) for bp in broker_positions)
                        if pe_sym and not pd.isna(pe_sym):
                            found_pe = any(self._is_symbol_in_pos(pe_sym, bp) for bp in broker_positions)
                    except Exception:
                        found_ce = found_pe = False

                if not found_ce and not found_pe:
                    if ALLOW_REENTRY_AFTER_STOP:
                        logger.info("No broker positions detected -> resetting to WAITING")
                        self._reset_all_positions()
                        self.state = "WAITING"
                        self._log_state("WAITING", "Reset - no broker positions found")
                        return True
                    else:
                        logger.info("No broker positions detected -> remain STOPPED_OUT")
                        self._reset_all_positions()
                        self.state = "STOPPED_OUT"
                        self._log_state("STOPPED_OUT", "No broker positions - staying STOPPED_OUT")
                        return True

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
                        self.positions[leg]['real_entered'] = bool(last.get(f'{leg}_real_entered', False))
                        
                        # Get token from symbol if missing
                        if not self.positions[leg].get('token'):
                            token = self._get_token_from_symbol(self.positions[leg]['symbol'])
                            if token:
                                self.positions[leg]['token'] = token
                        
                        logger.info(f"Recovered {leg.upper()} main: {self.positions[leg]['symbol']}")
                except Exception as e:
                    logger.debug(f"Failed to recover main leg {leg} from file: {e}")

                try:
                    hed_sym = last.get(f'{leg}_hedge_symbol', None)
                    if pd.notna(hed_sym) and str(hed_sym).strip():
                        self.hedges[leg]['symbol'] = str(hed_sym).strip()
                        self.hedges[leg]['entry_price'] = float(last.get(f'{leg}_hedge_entry', 0.0))
                        
                        # Get token from symbol if missing
                        if not self.hedges[leg].get('token'):
                            token = self._get_token_from_symbol(self.hedges[leg]['symbol'])
                            if token:
                                self.hedges[leg]['token'] = token
                                
                        logger.info(f"Recovered hedge {leg.upper()}: {self.hedges[leg]['symbol']}")
                except Exception as e:
                    logger.debug(f"Failed to recover hedge for {leg}: {e}")

            self._log_state("RECOVERED", f"Successfully recovered state: {self.state}")
            return True

        except Exception as e:
            logger.error(f"State file recovery failed: {e}")
            self._log_state("ERROR", f"Recovery failed: {e}")
            return False

    def _is_symbol_in_pos(self, symbol: Optional[str], broker_pos: Dict[str, Any]) -> bool:
        """Enhanced symbol matching for broker positions"""
        if not symbol or not broker_pos:
            return False
        try:
            norm_symbol = str(symbol).strip().upper()
            
            # Check various possible field names for symbol
            symbol_fields = ['tsym', 'tradingsymbol', 'TradingSymbol', 'symbol', 'Trading_Symbol']
            for field in symbol_fields:
                if field in broker_pos and str(broker_pos[field]).strip().upper() == norm_symbol:
                    return True
            
            # Check token-based matching
            if 'token' in broker_pos and self.positions['ce'].get('token'):
                if str(broker_pos['token']) == str(self.positions['ce'].get('token')):
                    return True
                    
            return False
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
                    if net_qty < 0:  # Only short positions for IBBM
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
        """Register positions with PositionManager - CRITICAL FOR MAPPING"""
        try:
            target_manager = position_manager or self.position_manager
            if not target_manager:
                logger.error("No position manager available for registration")
                return False

            spot_price = self._get_current_spot_price()
            current_time = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
            
            # Register main positions
            for leg in ['ce', 'pe']:
                pos = self.positions[leg]
                if pos['symbol'] and pos['token']:
                    key = f"{pos['symbol']}_{pos['token']}"
                    target_manager._strategy_symbol_token_map[key] = {
                        'strategy_name': STRATEGY_NAME,
                        'spot_price': spot_price,
                        'timestamp': current_time,
                        'leg_type': leg.upper(),
                        'position_type': 'MAIN'
                    }
                    logger.info(f"Registered {leg.upper()} main with PositionManager: {pos['symbol']}")
                    
                hedge = self.hedges[leg]
                if hedge['symbol'] and hedge['token']:
                    key = f"{hedge['symbol']}_{hedge['token']}"
                    target_manager._strategy_symbol_token_map[key] = {
                        'strategy_name': f"{STRATEGY_NAME}",
                        'spot_price': spot_price,
                        'timestamp': current_time,
                        'leg_type': leg.upper(),
                        'position_type': 'HEDGE'
                    }
                    logger.info(f"Registered {leg.upper()} hedge with PositionManager: {hedge['symbol']}")
            
            # Ensure the mapping is saved
            target_manager._save_strategy_mapping()
            logger.info("Successfully registered IBBM strategy positions with PositionManager")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register strategy positions: {str(e)}")
            return False
        
    def execute_strategy(self):
        """Public strategy execution method"""
        logger.info("Executing IBBM strategy")
        self._run_entry_cycle()

    # ===== Core Strategy Logic =====
    def _check_and_execute_strategy(self):
        current_time = ISTTimeUtils.current_time()

        # 1. End of day exit logic
        # if (current_time >= EOD_EXIT_TIME and self.state in ['VIRTUAL', 'WAIT_FOR_MAIN', 'ACTIVE']):
        #     self._exit_all_positions(reason="End of trading day")
        #     return

        # 2. Strategy entry logic
        if (TRADING_START_TIME <= current_time <= TRADING_END_TIME and 
            self.state == "WAITING" and
            current_time.minute in ENTRY_MINUTES):
            
            logger.info("Entry window detected - starting entry cycle")
            self._run_entry_cycle()

    def _run_entry_cycle(self):
        expiry_date = self._get_current_expiry()
        if not expiry_date:
            logger.error("No expiry available for entry cycle.")
            return

        ok = self._setup_virtual_positions(expiry_date)
        if ok:
            self.state = "VIRTUAL"
            self._log_state("VIRTUAL", "Virtual CE & PE created")
            self.register_strategy_positions()
            logger.info("Virtual CE & PE created. Now monitoring for virtual SL hit.")
        else:
            self.state = "WAIT_FOR_MAIN"
            self._log_state("WAIT_FOR_MAIN", "Waiting for main legs to satisfy premium range")
            logger.info("Could not create both virtual legs; state WAIT_FOR_MAIN.")

    def _setup_virtual_positions(self, expiry_date) -> bool:
        try:
            logger.info(f"Searching for main options in premium range {PREMIUM_RANGE} for expiry {expiry_date}")
            found = {}
            for opt_type, leg in [('CE', 'ce'), ('PE', 'pe')]:
                symbol, token, ltp = self._find_option_by_price(expiry_date, opt_type, PREMIUM_RANGE)
                if symbol:
                    initial_sl = round_sl(ltp * INITIAL_SL_MULTIPLIER)
                    self.positions[leg].update({
                        'symbol': symbol, 'token': token, 'entry_price': ltp,
                        'ltp': ltp, 'initial_sl': initial_sl, 'current_sl': initial_sl,
                        'max_profit_price': ltp, 'trailing_step': 0, 'real_entered': False, 'sl_hit': False
                    })
                    logger.info(f"Virtual {leg.upper()} prepared: {symbol} @ {ltp}, SL={initial_sl}")
                    found[leg] = True
                else:
                    logger.debug(f"No {opt_type} found in premium range yet.")
                    found[leg] = False
            return bool(found.get('ce') and found.get('pe'))
        except Exception as e:
            logger.error(f"_setup_virtual_positions failed: {e}\n{traceback.format_exc()}")
            return False

    def _monitor_all(self):
        now = ISTTimeUtils.current_time()
        if not (TRADING_START_TIME <= now <= TRADING_END_TIME):
            return

        if self.state == "WAIT_FOR_MAIN":
            expiry_date = self._get_current_expiry()
            if expiry_date:
                ok = self._setup_virtual_positions(expiry_date)
                if ok:
                    self.state = "VIRTUAL"
                    self._log_state("VIRTUAL", "Main legs found during WAIT_FOR_MAIN")
                    logger.info("Main legs found during WAIT_FOR_MAIN; state -> VIRTUAL")

        elif self.state == "VIRTUAL":
            self._monitor_virtual_sl()

        elif self.state == "ACTIVE":
            self._monitor_active_positions()
            self._validate_active_positions_exist()

    def _monitor_virtual_sl(self):
        try:
            for leg in ['ce', 'pe']:
                pos = self.positions[leg]
                if not pos['symbol']:
                    continue
                ltp = self._get_option_ltp(pos['symbol'])
                if ltp is None:
                    continue
                pos['ltp'] = ltp
                logger.debug(f"Virtual {leg.upper()} LTP={ltp} SL={pos['current_sl']}")
                if not pos.get('sl_hit', False) and ltp >= pos.get('current_sl', float('inf')):
                    logger.info(f"Virtual {leg.upper()} SL hit: LTP={ltp} >= SL={pos.get('current_sl')} - preparing to activate opposite side real position")
                    opposite_leg = 'pe' if leg == 'ce' else 'ce'
                    self._activate_opposite_real_position(opposite_leg)
                    return
        except Exception as e:
            logger.error(f"_monitor_virtual_sl failed: {e}\n{traceback.format_exc()}")

    def _activate_opposite_real_position(self, leg: str) -> bool:
        try:
            expiry_date = self._get_current_expiry()
            if not expiry_date:
                logger.error("No expiry available; cannot activate real position.")
                return False

            main_symbol, main_token, main_ltp = self._find_option_by_price(expiry_date, leg.upper(), PREMIUM_RANGE)
            if not main_symbol:
                logger.warning(f"No fresh main {leg.upper()} found in PREMIUM_RANGE {PREMIUM_RANGE} - abort activation")
                return False

            hedge_symbol, hedge_token, hedge_ltp = self._find_option_by_price(expiry_date, leg.upper(), HEDGE_RANGE)
            if not hedge_symbol:
                logger.warning(f"No hedge {leg.upper()} found in HEDGE_RANGE {HEDGE_RANGE} - abort activation")
                return False

            logger.info(f"Activation plan for {leg.upper()}: hedge={hedge_symbol}@{hedge_ltp}, main(new)={main_symbol}@{main_ltp}")

            hedge_ok = self._place_order(hedge_symbol, hedge_token, 'BUY')
            sell_ok = False
            if hedge_ok:
                logger.info(f"Hedge BUY placed for {leg.upper()} ({hedge_symbol})")
                time.sleep(2)  # Wait for hedge order to execute
                sell_ok = self._place_order(main_symbol, main_token, 'SELL')
                if sell_ok:
                    logger.info(f"Main SELL placed for {leg.upper()} ({main_symbol})")
                else:
                    logger.error(f"Main SELL failed for {leg.upper()} ({main_symbol}) after hedge placed. Attempting to unwind hedge.")
                    try:
                        unwind_ok = self._place_order(hedge_symbol, hedge_token, 'SELL')
                        if unwind_ok:
                            logger.info(f"Unwound hedge {leg.upper()} ({hedge_symbol}) after failed main SELL.")
                        else:
                            logger.error(f"Failed to unwind hedge {hedge_symbol} after failed main SELL. Manual intervention may be required.")
                    except Exception as e:
                        logger.exception(f"Exception while trying to unwind hedge: {e}")

            if hedge_ok and sell_ok:
                self.hedges[leg].update({'symbol': hedge_symbol, 'token': hedge_token, 'entry_price': hedge_ltp, 'ltp': hedge_ltp})
                new_sl = round_sl(main_ltp * INITIAL_SL_MULTIPLIER)
                self.positions[leg].update({
                    'symbol': main_symbol, 'token': main_token, 'entry_price': main_ltp, 'ltp': main_ltp,
                    'initial_sl': new_sl, 'current_sl': new_sl, 'max_profit_price': main_ltp, 'trailing_step': 0, 'real_entered': True, 'sl_hit': False
                })
                self.state = "ACTIVE"
                self._log_state("ACTIVE", f"Activated real {leg.upper()} with hedge + main SELL", activated_leg=leg.upper(), main_symbol=main_symbol, hedge_symbol=hedge_symbol)
                self.register_strategy_positions()  # Re-register with new positions
                logger.info(f"Activation successful for {leg.upper()}: hedge and main SELL placed.")
                return True
            else:
                logger.warning(f"Activation incomplete: hedge_ok={hedge_ok} sell_ok={sell_ok}")
                return False

        except Exception as e:
            logger.error(f"_activate_opposite_real_position exception: {e}\n{traceback.format_exc()}")
            return False

    def _monitor_active_positions(self):
        try:
            for leg in ['ce', 'pe']:
                pos = self.positions[leg]
                if not pos['symbol'] or not pos.get('real_entered', False):
                    continue
                ltp = self._get_option_ltp(pos['symbol'])
                if ltp is None:
                    logger.debug(f"No LTP for active {leg.upper()} ({pos['symbol']}) - skipping SL check")
                    continue
                pos['ltp'] = ltp

                if pos.get('max_profit_price', 0.0) == 0.0:
                    pos['max_profit_price'] = pos['entry_price']
                if ltp < pos.get('max_profit_price', pos['entry_price']):
                    pos['max_profit_price'] = ltp

                self._update_trailing_sl(leg)

                current_sl = pos.get('current_sl', None)
                if current_sl and not pos.get('sl_hit', False) and ltp >= current_sl:
                    logger.info(f"Real {leg.upper()} SL hit: LTP={ltp} >= SL={current_sl} — exiting position")
                    self._exit_real_position(leg, ltp)

        except Exception as e:
            logger.error(f"_monitor_active_positions failed: {e}\n{traceback.format_exc()}")

    def _update_trailing_sl(self, leg: str):
        try:
            pos = self.positions[leg]
            if not pos['symbol'] or pos['entry_price'] <= 0:
                return
            entry_price = pos['entry_price']
            max_profit_price = pos.get('max_profit_price', entry_price)
            profit_frac = (entry_price - max_profit_price) / entry_price
            new_step = 0
            for i, threshold in enumerate(TRAILING_SL_STEPS):
                if profit_frac >= threshold:
                    new_step = i + 1
            current_step = pos.get('trailing_step', 0)
            if new_step > current_step:
                sl_multiplier = 1.0 - (0.05 * new_step)
                new_sl = round_sl(entry_price * sl_multiplier)
                pos['current_sl'] = new_sl
                pos['trailing_step'] = new_step
                logger.info(f"Trailing SL updated for {leg.upper()} to {new_sl} at step {new_step}")
                self._log_state("ACTIVE", f"{leg.upper()} trailing SL updated to {new_sl}", trailing_step=new_step)
        except Exception as e:
            logger.error(f"_update_trailing_sl error for {leg}: {e}\n{traceback.format_exc()}")

    def _exit_real_position(self, leg: str, ltp: float):
        try:
            pos = self.positions[leg]
            if not pos['symbol']:
                logger.warning(f"No real {leg} position to exit")
                return False
            ok = self._place_order(pos['symbol'], pos['token'], 'BUY')
            if ok:
                pos['sl_hit'] = True
                pos['real_entered'] = False
                logger.info(f"{leg.upper()} real position exited @ {ltp}")
                self._log_state("STOPPED_OUT", f"{leg.upper()} SL hit at {ltp}", leg=leg.upper(), exit_ltp=ltp)
                both_closed = all(not self.positions[l]['real_entered'] for l in ['ce', 'pe'])
                if both_closed:
                    self.state = "STOPPED_OUT"
                    logger.info("Both real legs closed -> STATE STOPPED_OUT")
                return True
            else:
                logger.error(f"Failed to exit real position {pos['symbol']} via BUY")
                return False
        except Exception as e:
            logger.error(f"_exit_real_position exception: {e}\n{traceback.format_exc()}")
            return False

    def _exit_all_positions(self, reason: str = ""):
        logger.info(f"Exiting all positions: {reason}")
        all_closed_successfully = True

        for leg in ['ce', 'pe']:
            pos = self.positions[leg]
            if pos['symbol'] and pos.get('real_entered', False):
                try:
                    ok = self._place_order(pos['symbol'], pos['token'], 'BUY')
                    if ok:
                        logger.info(f"Exited main {leg.upper()} {pos['symbol']}")
                        self.positions[leg] = self._empty_position()
                    else:
                        logger.error(f"Failed to exit main {leg.upper()} {pos['symbol']}")
                        all_closed_successfully = False
                except Exception as e:
                    logger.error(f"Exception exiting main {leg}: {e}")
                    all_closed_successfully = False

        for leg in ['ce', 'pe']:
            hedge = self.hedges[leg]
            if hedge['symbol']:
                try:
                    ok = self._place_order(hedge['symbol'], hedge['token'], 'SELL')
                    if ok:
                        logger.info(f"Exited hedge {leg.upper()} {hedge['symbol']}")
                        self.hedges[leg] = self._empty_position()
                    else:
                        logger.error(f"Failed to exit hedge {leg.upper()} {hedge['symbol']}")
                        all_closed_successfully = False
                except Exception as e:
                    logger.error(f"Exception exiting hedge {leg}: {e}")
                    all_closed_successfully = False

        if all_closed_successfully:
            self.state = "COMPLETED"
            self._log_state("COMPLETED", f"All closed: {reason}")
            logger.info("All positions closed successfully.")
        else:
            self._log_state(self.state, f"Exit attempted but some positions may remain: {reason}")
            logger.warning("Some positions may remain after attempted exit.")

    # ===== Broker / Quote / Order wrappers =====
    def _find_option_by_price(self, expiry_date, option_type: str, price_range: Tuple[float, float]) -> Tuple[Optional[str], Optional[str], Optional[float]]:
        try:
            if not os.path.exists(NFO_SYMBOLS_FILE):
                logger.error(f"{NFO_SYMBOLS_FILE} not found")
                return None, None, None

            expiry_str = expiry_date.strftime("%d-%b-%Y").upper()
            df = pd.read_csv(NFO_SYMBOLS_FILE)

            opts = df[
                (df['Instrument'].str.strip() == "OPTIDX") &
                (df['Symbol'].str.strip() == "NIFTY") &
                (df['Expiry'].str.strip().str.upper() == expiry_str) &
                (df['OptionType'].str.strip().str.upper() == option_type.upper())
            ]

            if opts.empty:
                logger.debug(f"No options in chain for {option_type} expiry {expiry_str}")
                return None, None, None

            for _, row in opts.iterrows():
                symbol = str(row.get('TradingSymbol', '')).strip()
                token = str(row.get('Token', '')).strip()
                if not symbol:
                    continue
                ltp = self._get_option_ltp(symbol)
                if ltp is None:
                    continue
                if price_range[0] <= ltp <= price_range[1]:
                    return symbol, token, ltp

            return None, None, None

        except Exception as e:
            logger.error(f"_find_option_by_price error: {e}\n{traceback.format_exc()}")
            return None, None, None

    def _get_option_ltp(self, symbol_or_token: str) -> Optional[float]:
        try:
            client = self._get_primary_client()
            if not client or not symbol_or_token:
                return None

            q = None
            try:
                q = client.get_quotes('NFO', symbol_or_token)
            except Exception:
                try:
                    q = client.get_quotes('NFO', symbol_or_token)
                except Exception:
                    q = None

            if not q:
                return None

            if isinstance(q, dict):
                if q.get('stat') in ['Ok', 'OK', 'ok'] and 'lp' in q and q.get('lp') is not None:
                    try:
                        return float(q.get('lp'))
                    except Exception:
                        pass
                if 'data' in q and isinstance(q['data'], dict):
                    if symbol_or_token in q['data'] and isinstance(q['data'][symbol_or_token], dict):
                        inner = q['data'][symbol_or_token]
                        if 'lp' in inner and inner.get('lp') is not None:
                            return float(inner.get('lp'))
                    for v in q['data'].values():
                        if isinstance(v, dict) and 'lp' in v and v.get('lp') is not None:
                            return float(v.get('lp'))

            return None

        except Exception as e:
            logger.debug(f"_get_option_ltp error for {symbol_or_token}: {e}")
            return None

    def _place_order(self, symbol: str, token: str, action: str) -> bool:
        try:
            client = self._get_primary_client()
            if not client:
                logger.error("No trading client available to place order")
                return False

            sh_action = 'B' if action.upper() == 'BUY' else 'S'
            try:
                order_res = client.place_order(
                    buy_or_sell=sh_action,
                    product_type=ORDER_PRODUCT_TYPE,
                    exchange=ORDER_EXCHANGE,
                    tradingsymbol=symbol,
                    quantity=LOT_SIZE,
                    discloseqty=0,
                    price_type='MKT',
                    price=0.0,
                    trigger_price=None,
                    retention='DAY',
                    remarks=f"{STRATEGY_NAME}_{action.upper()}_{symbol}"
                )
            except Exception as e:
                logger.error(f"Broker place_order raised exception: {e}")
                return False

            if not order_res:
                return False

            if isinstance(order_res, dict):
                if order_res.get('stat') in ['Ok', 'OK', 'ok']:
                    return True
                if order_res.get('status', '').lower() in ['success', 'ok']:
                    return True
                return True
            if isinstance(order_res, bool):
                return order_res
            return True

        except Exception as e:
            logger.error(f"_place_order failed: {e}\n{traceback.format_exc()}")
            return False

    def _get_broker_positions(self) -> List[Dict[str, Any]]:
        try:
            client = self._get_primary_client()
            if not client:
                return []
            res = client.get_positions()
            if not res:
                return []
            if isinstance(res, dict) and 'data' in res:
                data = res['data']
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    return list(data.values())
            if isinstance(res, list):
                return res
            return []
        except Exception as e:
            logger.warning(f"Failed to get broker positions: {e}")
            return []

    def _get_primary_client(self):
        try:
            if self.client_manager and hasattr(self.client_manager, "get_primary_client"):
                return self.client_manager.get_primary_client()
            elif self.client_manager and hasattr(self.client_manager, "clients") and self.client_manager.clients:
                return self.client_manager.clients[0][2]
            else:
                logger.debug("Primary client not available")
                return None
        except Exception as e:
            logger.debug(f"Error getting primary client: {e}")
            return None

    def _validate_active_positions_exist(self):
        try:
            if self.state != "ACTIVE":
                return
            broker_positions = self._get_broker_positions()
            found_ce = any(self._is_symbol_in_pos(self.positions['ce']['symbol'], bp) for bp in broker_positions) if self.positions['ce']['symbol'] else False
            found_pe = any(self._is_symbol_in_pos(self.positions['pe']['symbol'], bp) for bp in broker_positions) if self.positions['pe']['symbol'] else False

            if not found_ce and not found_pe:
                logger.info("ACTIVE state but no broker positions found for CE/PE.")
                if ALLOW_REENTRY_AFTER_STOP:
                    logger.info("ALLOW_REENTRY_AFTER_STOP True -> resetting to WAITING.")
                    self._reset_all_positions()
                    self.state = "WAITING"
                    self._log_state("WAITING", "Auto-reset: ACTIVE but no broker positions")
                else:
                    logger.info("ALLOW_REENTRY_AFTER_STOP False -> marking STOPPED_OUT.")
                    self._reset_all_positions()
                    self.state = "STOPPED_OUT"
                    self._log_state("STOPPED_OUT", "No broker positions found; staying STOPPED_OUT")
        except Exception as e:
            logger.error(f"_validate_active_positions_exist error: {e}\n{traceback.format_exc()}")

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

    def _get_current_spot_price(self):
        try:
            client = self._get_primary_client()
            if client:
                quote = client.get_quotes('NSE', '26000')
                
                if not quote or quote.get('stat') != 'Ok':
                    logger.error(f"Failed to get NIFTY quote: {quote}")
                    return 0.0
                    
                spot_price = float(quote.get('lp', 0))
                logger.info(f"Current NIFTY spot price: {spot_price}")
                return spot_price
                
            logger.error("No clients available for spot price")
            return 0.0
                
        except Exception as e:
            logger.error(f"Failed to get spot price: {str(e)}")
            return 0.0

    def _get_token_from_symbol(self, symbol: str) -> Optional[str]:
        """Get token from NFO symbols file"""
        try:
            if not os.path.exists(NFO_SYMBOLS_FILE):
                logger.error("NFO_symbols.txt file not found")
                return None
                
            df = pd.read_csv(NFO_SYMBOLS_FILE)
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

    # ===== State logging =====
    def _log_state(self, status: str, comments: str = ""):
        """Consistent state logging - REMOVE **extra parameter"""
        try:
            now = ISTTimeUtils.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Clean comments to prevent CSV issues
            cleaned_comments = str(comments).replace(',', ';').replace('\n', ' | ').replace('"', "'")
            
            # ALWAYS use exactly 16 columns
            row = {
                'timestamp': now,
                'status': status,
                'comments': cleaned_comments,
                'ce_symbol': self.positions['ce'].get('symbol') or '',
                'ce_entry_price': float(self.positions['ce'].get('entry_price', 0.0)),
                'ce_ltp': float(self.positions['ce'].get('ltp', 0.0)),
                'ce_sl_price': float(self.positions['ce'].get('current_sl', 0.0)),
                'ce_real_entered': bool(self.positions['ce'].get('real_entered', False)),
                'pe_symbol': self.positions['pe'].get('symbol') or '',
                'pe_entry_price': float(self.positions['pe'].get('entry_price', 0.0)),
                'pe_ltp': float(self.positions['pe'].get('ltp', 0.0)),
                'pe_sl_price': float(self.positions['pe'].get('current_sl', 0.0)),
                'pe_real_entered': bool(self.positions['pe'].get('real_entered', False)),
                'ce_hedge_symbol': self.hedges['ce'].get('symbol') or '',
                'ce_hedge_entry': float(self.hedges['ce'].get('entry_price', 0.0)),
                'pe_hedge_symbol': self.hedges['pe'].get('symbol') or '',
                'pe_hedge_entry': float(self.hedges['pe'].get('entry_price', 0.0))
            }
            
            df = pd.DataFrame([row])
            write_header = not os.path.exists(self.current_state_file)
            df.to_csv(self.current_state_file, mode='a', header=write_header, index=False)
            logger.debug(f"State logged: {status} - {cleaned_comments}")
            
        except Exception as e:
            logger.error(f"CRITICAL: State logging failed: {e}\n{traceback.format_exc()}")

    def cleanup(self):
        logger.info("Cleaning up IBBM strategy resources")
        try:
            if hasattr(self.strategy_timer, "isActive") and self.strategy_timer.isActive():
                self.strategy_timer.stop()
                logger.debug("Strategy timer stopped")
            if hasattr(self.monitor_timer, "isActive") and self.monitor_timer.isActive():
                self.monitor_timer.stop()
                logger.debug("Monitor timer stopped")
        except Exception:
            logger.exception("Error stopping strategy timers")

        self.state = "WAITING"
        self.positions = {'ce': self._empty_position(), 'pe': self._empty_position()}
        self.hedges = {'ce': self._empty_position(), 'pe': self._empty_position()}

        logger.info("Cleanup complete")