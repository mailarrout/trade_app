# strategy_ibbm.py
"""
IBBMStrategy - Expanded, production-ready module.

Behavior summary:
 - Virtual CE & PE created at entry (PREMIUM_RANGE)
 - No hedges placed at entry
 - When a virtual leg hits its SL:
     1) Re-scan the option chain for a fresh main symbol on the OPPOSITE leg in PREMIUM_RANGE
        - If not found -> ABORT (no hedge, no sell)
     2) Find hedge (HEDGE_RANGE) on the same opposite leg
        - If no hedge found -> ABORT (no hedge, no sell)
     3) Place hedge BUY (real) and then place main SELL (real) back-to-back
        - If hedge succeeds and SELL fails -> immediately exit hedge (SELL hedge) to avoid naked hedge
 - Trailing SL, EOD exit, state logging, detailed recovery
 - All configuration constants at top for easy editing
"""

# Standard libs
import os
import sys
import logging
import math
import traceback
from datetime import datetime, time, timedelta
from functools import wraps
from typing import Optional, Dict, Any, Tuple, List, Union
from PyQt5.QtCore import QTimer
# Third-party
import pandas as pd
from pytz import timezone

# ----------------------------- CONFIGURATION / CONSTANTS -----------------------------
IST = timezone('Asia/Kolkata')
LOGGER_NAME = __name__

# Strategy identity
STRATEGY_NAME = "IBBM Intraday"

# Trading windows (IST)
TRADING_START_TIME = time(9, 45)
TRADING_END_TIME = time(14, 45)
EOD_EXIT_TIME = time(15, 15)

# Entry minute tolerance (xx:15 and xx:45 +/-1 minute)
ENTRY_MINUTES = [14, 15, 16, 44, 45, 46]
MONITORING_MINUTES = [15, 45]

# Option selection ranges
STRIKE_RANGE = 1000                    # +/- near ATM if used in other logic
PREMIUM_RANGE = (70.0, 100.0)          # Desired premium band for main options (virtual & real SELL)
HEDGE_RANGE = (5.0, 15.0)              # Hedge price band for real BUYs when activating

# Stop-loss & trailing SL
INITIAL_SL_MULTIPLIER = 1.20  # 20% SL         
SL_ROUNDING_FACTOR = 20                # to round SL to nearest 1/20 (0.05)
TRAILING_SL_STEPS = [0.10, 0.20, 0.30, 0.40, 0.50]  # trailing thresholds (fractions of profit)

# Hedge search limits
HEDGE_MAX_SEARCH_DISTANCE = 1000       # max strike distance to search for hedge

# Timers (ms)
STRATEGY_CHECK_INTERVAL = 60_000       # 1 minute
MONITORING_INTERVAL = 10_000           # 10 seconds

# Files
NFO_SYMBOLS_FILE = "NFO_symbols.txt"   # expected CSV of option chain
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Order config
LOT_SIZE = 75
ORDER_PRODUCT_TYPE = 'M'
ORDER_EXCHANGE = 'NFO'

# Re-entry policy after STOPPED_OUT/ACTIVE with no broker positions
ALLOW_REENTRY_AFTER_STOP = True        # True -> auto-reset to WAITING if recovery shows no positions

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(LOGGER_NAME)

# -------------------------------------------------------------------------------

# ----------------------------- UTILS / DECORATORS --------------------------------
def safe_log(context: str):
    """Decorator to wrap functions with try/except and log exceptions."""
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
    def current_time() -> time:
        return ISTTimeUtils.now().time()

    @staticmethod
    def current_date_str() -> str:
        return ISTTimeUtils.now().strftime("%Y-%m-%d")


def round_sl(price: float) -> float:
    """Round SL to nearest tick defined by SL_ROUNDING_FACTOR."""
    return math.ceil(price * SL_ROUNDING_FACTOR) / SL_ROUNDING_FACTOR
# -------------------------------------------------------------------------------


# ----------------------------- STRATEGY CLASS ------------------------------------
class IBBMStrategy:
    """
    IBBMStrategy (expanded)

    Use:
      strategy = IBBMStrategy(ui, client_manager)
      - ui: optional object providing ExecuteStrategyQPushButton and ExpiryListDropDown
      - client_manager: object providing .clients with broker client at [0][2]
    """

    def __init__(self, ui: Optional[Any], client_manager: Optional[Any]):
        self.ui = ui
        self.client_manager = client_manager

        # state file path
        self.current_state_file = os.path.join(LOG_DIR, f"{ISTTimeUtils.current_date_str()}_ibbm_strategy_state.csv")

        # possible states: WAITING | WAIT_FOR_MAIN | VIRTUAL | ACTIVE | STOPPED_OUT | COMPLETED
        self.state: str = "WAITING"

        # positions - main legs (virtual or real), hedges separately
        self.positions: Dict[str, Dict[str, Any]] = {
            'ce': self._empty_position(),
            'pe': self._empty_position()
        }
        self.hedges: Dict[str, Dict[str, Any]] = {
            'ce': self._empty_position(),
            'pe': self._empty_position()
        }

        # internal flags
        self._positions_validated = False
        self._first_monitoring_logged = False

        # Setup timers (QTimer if available)
        self.strategy_timer = QTimer()
        try:
            self.strategy_timer.timeout.connect(self._check_and_execute_strategy)
            self.strategy_timer.start(STRATEGY_CHECK_INTERVAL)
            logger.debug("Strategy timer started.")
        except Exception:
            logger.debug("Strategy timer could not be started (QTimer not available).")

        self.monitor_timer = QTimer()
        try:
            self.monitor_timer.timeout.connect(self._monitor_all)
            self.monitor_timer.start(MONITORING_INTERVAL)
            logger.debug("Monitor timer started.")
        except Exception:
            logger.debug("Monitor timer could not be started (QTimer not available).")

        # bind UI button if present
        try:
            if hasattr(self.ui, 'ExecuteStrategyQPushButton'):
                self.ui.ExecuteStrategyQPushButton.clicked.connect(self.on_execute_strategy_clicked)
        except Exception:
            logger.debug("Could not bind UI execute button (UI may be absent).")

        # attempt recovery from state file on start
        self._try_recover_from_state_file()

        logger.info("IBBMStrategy initialized; state=%s", self.state)

    # -------------------- small helpers --------------------
    def _empty_position(self) -> Dict[str, Any]:
        return {
            'symbol': None, 'token': None, 'ltp': 0.0,
            'entry_price': 0.0, 'initial_sl': 0.0, 'current_sl': 0.0,
            'sl_hit': False, 'max_profit_price': 0.0, 'trailing_step': 0,
            'real_entered': False
        }

    # -------------------- Recovery & State --------------------
    @safe_log("recovery")
    def _try_recover_from_state_file(self):
        """
        Recover minimal state from today's CSV file.
        If last state was ACTIVE/STOPPED_OUT but broker shows no positions,
        auto-reset to WAITING if ALLOW_REENTRY_AFTER_STOP True, otherwise stay STOPPED_OUT.
        """
        if not os.path.exists(self.current_state_file):
            logger.info("No state file for today; starting fresh.")
            return

        try:
            df = pd.read_csv(self.current_state_file, on_bad_lines='skip')
        except Exception as e:
            logger.warning("Failed to read state file for recovery: %s", e)
            return

        if df.empty:
            logger.info("State file empty for today.")
            return

        last = df.iloc[-1]
        last_status = str(last.get('status', 'WAITING')).strip()
        logger.info("Recovered last status from file: %s", last_status)

        # If state is ACTIVE or STOPPED_OUT, check with broker to see if positions exist
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
                # no broker positions exist - decide based on ALLOW_REENTRY_AFTER_STOP
                if ALLOW_REENTRY_AFTER_STOP:
                    logger.info("No broker positions detected for recovered ACTIVE/STOPPED_OUT -> resetting to WAITING (ALLOW_REENTRY_AFTER_STOP=True)")
                    self._reset_all_positions()
                    self.state = "WAITING"
                    # do not remove the state file, just continue
                    return
                else:
                    logger.info("No broker positions detected -> remain STOPPED_OUT (ALLOW_REENTRY_AFTER_STOP=False)")
                    self._reset_all_positions()
                    self.state = "STOPPED_OUT"
                    return

        # otherwise try to recover fields for convenience
        self.state = last_status

        # recover stored fields per leg if present
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
                    logger.info("Recovered %s main: %s", leg.upper(), self.positions[leg]['symbol'])
            except Exception:
                logger.debug("Failed to recover main leg %s from file.", leg)

            try:
                hed_sym = last.get(f'{leg}_hedge_symbol', None)
                if pd.notna(hed_sym) and str(hed_sym).strip():
                    self.hedges[leg]['symbol'] = str(hed_sym).strip()
                    self.hedges[leg]['entry_price'] = float(last.get(f'{leg}_hedge_entry', 0.0))
                    logger.info("Recovered hedge %s: %s", leg.upper(), self.hedges[leg]['symbol'])
            except Exception:
                logger.debug("Failed to recover hedge for %s", leg)

    def _reset_all_positions(self):
        self.positions = {'ce': self._empty_position(), 'pe': self._empty_position()}
        self.hedges = {'ce': self._empty_position(), 'pe': self._empty_position()}

    def _is_symbol_in_pos(self, symbol: Optional[str], broker_pos: Dict[str, Any]) -> bool:
        """Check if symbol or token matches a broker pos entry across common fields."""
        if not symbol or not broker_pos:
            return False
        try:
            norm = str(symbol).strip().upper()
            for key in ['tradingsymbol', 'TradingSymbol', 'symbol', 'Trading_Symbol', 'instrument']:
                if key in broker_pos and str(broker_pos[key]).strip().upper() == norm:
                    return True
            # token matching
            if 'token' in broker_pos:
                if str(broker_pos.get('token')).strip() == str(broker_pos.get('Token', '')).strip():
                    return True
            # sometimes the client returns instrument_token or similar
            if 'instrumentToken' in broker_pos and str(broker_pos['instrumentToken']) == norm:
                return True
        except Exception:
            return False
        return False

    # -------------------- Manual UI trigger --------------------
    def on_execute_strategy_clicked(self):
        """Manual execution via UI button - allowed only at entry minutes and trading hours."""
        now = ISTTimeUtils.current_time()
        logger.info("Manual execute clicked at %s", now)
        if now.minute not in ENTRY_MINUTES:
            logger.warning("Manual execute allowed only in entry minutes.")
            return
        if not (TRADING_START_TIME <= now <= TRADING_END_TIME):
            logger.warning("Manual execute outside trading hours.")
            return
        if self.state not in ['WAITING']:
            logger.info("Manual execute ignored; current state=%s", self.state)
            return
        logger.info("Manual start accepted; running entry cycle.")
        self._run_entry_cycle()

    # -------------------- Periodic checks --------------------
    def _check_and_execute_strategy(self):
        """
        Called by strategy_timer periodically (every minute).
        Handles:
         - EOD forced exit
         - Entry window check
        """
        now = ISTTimeUtils.current_time()

        # EOD exit - force-close any open legs/hedges at EOD_EXIT_TIME
        if (now.hour == EOD_EXIT_TIME.hour and now.minute == EOD_EXIT_TIME.minute) and self.state in ['VIRTUAL', 'WAIT_FOR_MAIN', 'ACTIVE']:
            logger.info("EOD exit time hit - attempting to exit all positions")
            self._exit_all_positions(reason="EOD Exit")
            return

        # Entry logic: only start entry cycle in WAITING and inside trading hours at allowed minute
        if self.state == "WAITING" and TRADING_START_TIME <= now <= TRADING_END_TIME and now.minute in ENTRY_MINUTES:
            logger.info("Entry window detected - starting entry cycle")
            self._run_entry_cycle()

    def _run_entry_cycle(self):
        """
        Main entry cycle:
         - Determine expiry
         - Create virtual CE and PE in PREMIUM_RANGE (not placing real orders)
         - If not both found, set WAIT_FOR_MAIN (hedges not placed).
        """
        expiry_date = self._get_current_expiry()
        if not expiry_date:
            logger.error("No expiry available for entry cycle.")
            return

        # 1) Setup virtual main positions
        ok = self._setup_virtual_positions(expiry_date)
        if ok:
            self.state = "VIRTUAL"
            self._log_state("VIRTUAL", "Virtual CE & PE created")
            logger.info("Virtual CE & PE created. Now monitoring for virtual SL hit.")
        else:
            # If not found both, remain WAIT_FOR_MAIN, hedges not placed
            self.state = "WAIT_FOR_MAIN"
            self._log_state("WAIT_FOR_MAIN", "Hedges not taken; waiting for main legs to satisfy premium range")
            logger.info("Could not create both virtual legs; state WAIT_FOR_MAIN.")

    # -------------------- Virtual setup --------------------
    def _setup_virtual_positions(self, expiry_date) -> bool:
        """
        Find CE & PE in PREMIUM_RANGE and set them as virtual SELL legs (internal only).
        Returns True when both CE & PE virtual legs are present.
        """
        try:
            logger.info("Searching for main options in premium range %s for expiry %s", PREMIUM_RANGE, expiry_date)
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
                    logger.info("Virtual %s prepared: %s @ %s, SL=%s", leg.upper(), symbol, ltp, initial_sl)
                    found[leg] = True
                else:
                    logger.debug("No %s found in premium range yet.", opt_type)
                    found[leg] = False
            return bool(found.get('ce') and found.get('pe'))
        except Exception as e:
            logger.error("_setup_virtual_positions failed: %s\n%s", e, traceback.format_exc())
            return False

    # -------------------- Monitor --------------------
    def _monitor_all(self):
        """
        Frequent monitoring loop invoked by monitor_timer (~every 10 seconds).
        - Checks virtual SL triggers
        - If ACTIVE, checks real positions and trailing SL
        - If WAIT_FOR_MAIN, attempts to find main legs again
        """
        now = ISTTimeUtils.current_time()
        # If outside trading hours skip most checks; let EOD handle exit in main timer
        if not (TRADING_START_TIME <= now <= TRADING_END_TIME):
            return

        if self.state == "WAIT_FOR_MAIN":
            # attempt to find main legs repeatedly while hedges not taken
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
            # monitor active real positions for SL/trailing
            self._monitor_active_positions()
            # also validate broker positions existence and reconcile
            self._validate_active_positions_exist()

    def _monitor_virtual_sl(self):
        """
        Examine LTP for virtual legs. If LTP >= current SL for any virtual leg:
         - determine opposite leg (if PE virtual hit -> opposite CE)
         - attempt to activate opposite side real position using _activate_opposite_real_position
        """
        try:
            for leg in ['ce', 'pe']:
                pos = self.positions[leg]
                if not pos['symbol']:
                    continue
                ltp = self._get_option_ltp(pos['symbol'])
                if ltp is None:
                    continue
                pos['ltp'] = ltp
                logger.debug("Virtual %s LTP=%s SL=%s", leg.upper(), ltp, pos['current_sl'])
                if not pos.get('sl_hit', False) and ltp >= pos.get('current_sl', float('inf')):
                    logger.info("Virtual %s SL hit: LTP=%s >= SL=%s - preparing to activate opposite side real position", leg.upper(), ltp, pos.get('current_sl'))
                    opposite_leg = 'pe' if leg == 'ce' else 'ce'
                    # Activation will re-scan for fresh main symbol in PREMIUM_RANGE, find hedge in HEDGE_RANGE,
                    # and place hedge BUY & main SELL in a safe paired manner.
                    self._activate_opposite_real_position(opposite_leg)
                    # once we handled one virtual SL, stop processing until next monitor tick
                    return
        except Exception as e:
            logger.error("_monitor_virtual_sl failed: %s\n%s", e, traceback.format_exc())

    # -------------------- Activation: re-scan + hedge + sell --------------------
    def _activate_opposite_real_position(self, leg: str) -> bool:
        """
        When one virtual leg's SL triggers, activate opposite-side real position.
        Steps:
            1) Re-scan for a fresh main symbol for 'leg' in PREMIUM_RANGE.
               If not found -> abort.
            2) Find hedge for 'leg' in HEDGE_RANGE (5-15).
               If not found -> abort.
            3) Place hedge BUY then place main SELL back-to-back.
               If SELL fails after hedge placed -> exit hedge immediately (SELL hedge).
        Returns True if activation succeeded (hedge placed & main sell placed), False otherwise.
        """
        try:
            expiry_date = self._get_current_expiry()
            if not expiry_date:
                logger.error("No expiry available; cannot activate real position.")
                return False

            # Step 1: Re-scan for fresh main (70-100) for this leg
            main_symbol, main_token, main_ltp = self._find_option_by_price(expiry_date, leg.upper(), PREMIUM_RANGE)
            if not main_symbol:
                logger.warning("No fresh main %s found in PREMIUM_RANGE %s - abort activation", leg.upper(), PREMIUM_RANGE)
                # Do NOT place hedge if no main is available
                return False

            # Step 2: Find hedge (5-15) on same leg
            hedge_symbol, hedge_token, hedge_ltp = self._find_option_by_price(expiry_date, leg.upper(), HEDGE_RANGE)
            if not hedge_symbol:
                logger.warning("No hedge %s found in HEDGE_RANGE %s - abort activation", leg.upper(), HEDGE_RANGE)
                return False

            logger.info("Activation plan for %s: hedge=%s@%s, main(new)=%s@%s", leg.upper(), hedge_symbol, hedge_ltp, main_symbol, main_ltp)

            # Step 3: Place orders back-to-back: hedge BUY then main SELL
            hedge_ok = self._place_order(hedge_symbol, hedge_token, 'BUY')
            sell_ok = False
            if hedge_ok:
                logger.info("Hedge BUY placed for %s (%s)", leg.upper(), hedge_symbol)
                # place main SELL
                sell_ok = self._place_order(main_symbol, main_token, 'SELL')
                if sell_ok:
                    logger.info("Main SELL placed for %s (%s)", leg.upper(), main_symbol)
                else:
                    logger.error("Main SELL failed for %s (%s) after hedge placed. Attempting to unwind hedge.", leg.upper(), main_symbol)
                    # unwind hedge to avoid naked hedge
                    try:
                        unwind_ok = self._place_order(hedge_symbol, hedge_token, 'SELL')
                        if unwind_ok:
                            logger.info("Unwound hedge %s (%s) after failed main SELL.", leg.upper(), hedge_symbol)
                        else:
                            logger.error("Failed to unwind hedge %s after failed main SELL. Manual intervention may be required.", hedge_symbol)
                    except Exception as e:
                        logger.exception("Exception while trying to unwind hedge: %s", e)
            else:
                logger.error("Hedge BUY failed for %s (%s) - activation aborted", leg.upper(), hedge_symbol)

            # record results and update positions if both succeeded
            if hedge_ok and sell_ok:
                # update hedge record
                self.hedges[leg].update({'symbol': hedge_symbol, 'token': hedge_token, 'entry_price': hedge_ltp, 'ltp': hedge_ltp})
                # update main record to reflect fresh main symbol
                new_sl = round_sl(main_ltp * INITIAL_SL_MULTIPLIER)
                self.positions[leg].update({
                    'symbol': main_symbol, 'token': main_token, 'entry_price': main_ltp, 'ltp': main_ltp,
                    'initial_sl': new_sl, 'current_sl': new_sl, 'max_profit_price': main_ltp, 'trailing_step': 0, 'real_entered': True, 'sl_hit': False
                })
                self.state = "ACTIVE"
                self._log_state("ACTIVE", f"Activated real {leg.upper()} with hedge + main SELL", activated_leg=leg.upper(), main_symbol=main_symbol, hedge_symbol=hedge_symbol)
                logger.info("Activation successful for %s: hedge and main SELL placed.", leg.upper())
                return True
            else:
                logger.warning("Activation incomplete: hedge_ok=%s sell_ok=%s", hedge_ok, sell_ok)
                return False

        except Exception as e:
            logger.error("_activate_opposite_real_position exception: %s\n%s", e, traceback.format_exc())
            return False

    # -------------------- Active monitoring --------------------
    def _monitor_active_positions(self):
        """
        Monitor real-entered positions (real_entered True) for SL/trailing logic.
        Exits position if SL hit, updates trailing SL when profit accrues.
        """
        try:
            for leg in ['ce', 'pe']:
                pos = self.positions[leg]
                if not pos['symbol'] or not pos.get('real_entered', False):
                    continue
                ltp = self._get_option_ltp(pos['symbol'])
                if ltp is None:
                    logger.debug("No LTP for active %s (%s) - skipping SL check", leg.upper(), pos['symbol'])
                    continue
                pos['ltp'] = ltp

                # Update max_profit_price for short position (profit when LTP falls)
                if pos.get('max_profit_price', 0.0) == 0.0:
                    pos['max_profit_price'] = pos['entry_price']
                if ltp < pos.get('max_profit_price', pos['entry_price']):
                    pos['max_profit_price'] = ltp

                # trailing SL updates
                self._update_trailing_sl(leg)

                current_sl = pos.get('current_sl', None)
                if current_sl and not pos.get('sl_hit', False) and ltp >= current_sl:
                    logger.info("Real %s SL hit: LTP=%s >= SL=%s — exiting position", leg.upper(), ltp, current_sl)
                    self._exit_real_position(leg, ltp)

        except Exception as e:
            logger.error("_monitor_active_positions failed: %s\n%s", e, traceback.format_exc())

    def _update_trailing_sl(self, leg: str):
        """
        Update trailing SL for a short position based on profit thresholds.
        For short positions, profit increases as LTP decreases.
        """
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
                logger.info("Trailing SL updated for %s to %s at step %s", leg.upper(), new_sl, new_step)
                self._log_state("ACTIVE", f"{leg.upper()} trailing SL updated to {new_sl}", trailing_step=new_step)
        except Exception as e:
            logger.error("_update_trailing_sl error for %s: %s\n%s", leg, e, traceback.format_exc())

    def _exit_real_position(self, leg: str, ltp: float):
        """
        Exit only the real main SELL position (buy to close). Hedge remains unless user chooses to close it.
        """
        try:
            pos = self.positions[leg]
            if not pos['symbol']:
                logger.warning("No real %s position to exit", leg)
                return False
            ok = self._place_order(pos['symbol'], pos['token'], 'BUY')
            if ok:
                pos['sl_hit'] = True
                pos['real_entered'] = False
                logger.info("%s real position exited @ %s", leg.upper(), ltp)
                self._log_state("STOPPED_OUT", f"{leg.upper()} SL hit at {ltp}", leg=leg.upper(), exit_ltp=ltp)
                # if both real legs closed -> state STOPPED_OUT
                both_closed = all(not self.positions[l]['real_entered'] for l in ['ce', 'pe'])
                if both_closed:
                    self.state = "STOPPED_OUT"
                    logger.info("Both real legs closed -> STATE STOPPED_OUT")
                return True
            else:
                logger.error("Failed to exit real position %s via BUY", pos['symbol'])
                return False
        except Exception as e:
            logger.error("_exit_real_position exception: %s\n%s", e, traceback.format_exc())
            return False

    # -------------------- Exit all positions (EOD / manual) --------------------
    def _exit_all_positions(self, reason: str = ""):
        """
        Exit every open main and hedge position. Used at EOD or for final shutdown.
        Main real positions are bought to close; hedges are sold to close.
        """
        logger.info("Exiting all positions: %s", reason)
        all_closed_successfully = True

        # exit main legs (BUY)
        for leg in ['ce', 'pe']:
            pos = self.positions[leg]
            if pos['symbol'] and pos.get('real_entered', False):
                try:
                    ok = self._place_order(pos['symbol'], pos['token'], 'BUY')
                    if ok:
                        logger.info("Exited main %s %s", leg.upper(), pos['symbol'])
                        self.positions[leg] = self._empty_position()
                    else:
                        logger.error("Failed to exit main %s %s", leg.upper(), pos['symbol'])
                        all_closed_successfully = False
                except Exception as e:
                    logger.error("Exception exiting main %s: %s", leg, e)
                    all_closed_successfully = False

        # exit hedges (SELL)
        for leg in ['ce', 'pe']:
            hedge = self.hedges[leg]
            if hedge['symbol']:
                try:
                    ok = self._place_order(hedge['symbol'], hedge['token'], 'SELL')
                    if ok:
                        logger.info("Exited hedge %s %s", leg.upper(), hedge['symbol'])
                        self.hedges[leg] = self._empty_position()
                    else:
                        logger.error("Failed to exit hedge %s %s", leg.upper(), hedge['symbol'])
                        all_closed_successfully = False
                except Exception as e:
                    logger.error("Exception exiting hedge %s: %s", leg, e)
                    all_closed_successfully = False

        if all_closed_successfully:
            self.state = "COMPLETED"
            self._log_state("COMPLETED", f"All closed: {reason}")
            logger.info("All positions closed successfully.")
        else:
            self._log_state(self.state, f"Exit attempted but some positions may remain: {reason}")
            logger.warning("Some positions may remain after attempted exit.")

    # -------------------- Broker / Quote / Order wrappers --------------------
    def _find_option_by_price(self, expiry_date, option_type: str, price_range: Tuple[float, float]) -> Tuple[Optional[str], Optional[str], Optional[float]]:
        """
        Search NFO_SYMBOLS_FILE for a symbol matching option_type and expiry, whose LTP is within price_range.
        Returns (symbol, token, ltp) or (None, None, None)
        """
        try:
            if not os.path.exists(NFO_SYMBOLS_FILE):
                logger.error("%s not found", NFO_SYMBOLS_FILE)
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
                logger.debug("No options in chain for %s expiry %s", option_type, expiry_str)
                return None, None, None

            # iterate options and find first with LTP in range
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
            logger.error("_find_option_by_price error: %s\n%s", e, traceback.format_exc())
            return None, None, None

    def _get_option_ltp(self, symbol_or_token: str) -> Optional[float]:
        """
        Fetch LTP from broker get_quotes. Handle multiple shapes: direct lp, nested data, etc.
        Return float or None.
        """
        try:
            client = self._get_primary_client()
            if not client or not symbol_or_token:
                return None

            q = None
            try:
                q = client.get_quotes('NFO', symbol_or_token)
            except Exception:
                # Some clients might require token or symbol; try as best-effort
                try:
                    q = client.get_quotes('NFO', symbol_or_token)
                except Exception:
                    q = None

            if not q:
                return None

            # Shape 1: {'stat':'Ok', 'lp': 12.34}
            if isinstance(q, dict):
                if q.get('stat') in ['Ok', 'OK', 'ok'] and 'lp' in q and q.get('lp') is not None:
                    try:
                        return float(q.get('lp'))
                    except Exception:
                        pass
                # Shape 2: {'data': {'TOKEN': {'lp': x, ...}}}
                if 'data' in q and isinstance(q['data'], dict):
                    if symbol_or_token in q['data'] and isinstance(q['data'][symbol_or_token], dict):
                        inner = q['data'][symbol_or_token]
                        if 'lp' in inner and inner.get('lp') is not None:
                            return float(inner.get('lp'))
                    # fallback: pick first nested entry with lp
                    for v in q['data'].values():
                        if isinstance(v, dict) and 'lp' in v and v.get('lp') is not None:
                            return float(v.get('lp'))

            return None

        except Exception as e:
            logger.debug("_get_option_ltp error for %s: %s", symbol_or_token, e)
            return None

    def _place_order(self, symbol: str, token: str, action: str) -> bool:
        """
        Place a market order. For Shoonya-like API: buy_or_sell 'B' or 'S'.
        Accept different success shapes in response.
        Returns True on success, False on failure.
        """
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
                logger.error("Broker place_order raised exception: %s", e)
                return False

            if not order_res:
                return False

            # Interpret common success shapes
            if isinstance(order_res, dict):
                if order_res.get('stat') in ['Ok', 'OK', 'ok']:
                    return True
                if order_res.get('status', '').lower() in ['success', 'ok']:
                    return True
                # Some clients return nested 'data' => accept as success
                return True
            if isinstance(order_res, bool):
                return order_res
            # default: treat truthy response as success
            return True

        except Exception as e:
            logger.error("_place_order failed: %s\n%s", e, traceback.format_exc())
            return False

    def _get_broker_positions(self) -> List[Dict[str, Any]]:
        """
        Get positions from broker client and return as a list of dicts, tolerant to shapes.
        """
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
                    # convert dict values to list
                    return list(data.values())
            if isinstance(res, list):
                return res
            return []
        except Exception as e:
            logger.warning("Failed to get broker positions: %s", e)
            return []



    def _get_primary_client(self):
        """Return primary trading client if configured and accessible."""
        try:
            if self.client_manager and hasattr(self.client_manager, "clients") and self.client_manager.clients:
                return self.client_manager.clients[0][2]
        except Exception:
            logger.exception("Error fetching primary client")
        return None

    # -------------------- Validation & reconciliation --------------------
    def _validate_active_positions_exist(self):
        """
        If we are in ACTIVE but broker shows no positions for our main legs,
        reset to WAITING or STOPPED_OUT based on ALLOW_REENTRY_AFTER_STOP.
        """
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
            logger.error("_validate_active_positions_exist error: %s\n%s", e, traceback.format_exc())

    # -------------------- Expiry helper --------------------
    def _get_current_expiry(self):
        """
        Get current expiry from UI dropdown. The UI dropdown is expected to have entries in 'DD-Mon-YYYY'.
        If UI is absent, returns None (caller should provide expiry in that case).
        """
        try:
            if not self.ui or not hasattr(self.ui, 'ExpiryListDropDown'):
                logger.debug("ExpiryListDropDown not available in UI.")
                return None
            expiry_dates = []
            try:
                for i in range(self.ui.ExpiryListDropDown.count()):
                    txt = self.ui.ExpiryListDropDown.itemText(i)
                    try:
                        d = datetime.strptime(txt, "%d-%b-%Y").date()
                        expiry_dates.append(d)
                    except Exception:
                        logger.debug("Could not parse expiry dropdown item: %s", txt)
                expiry_dates.sort()
                today = datetime.now().date()
                for d in expiry_dates:
                    if d >= today:
                        return d
                return expiry_dates[-1] if expiry_dates else None
            except Exception as e:
                logger.exception("_get_current_expiry encountered an exception: %s", e)
                return None
        except Exception:
            return None

    # -------------------- State logging --------------------
    def _log_state(self, status: str, comments: str = "", **extra):
        """
        Append a detailed row to today's state CSV for auditing & recovery.
        Fields captured include both main legs, hedges, and any extra metadata.
        """
        try:
            now = ISTTimeUtils.now().strftime("%Y-%m-%d %H:%M:%S")
            row = {
                'timestamp': now,
                'status': status,
                'comments': comments,
                # main CE
                'ce_symbol': self.positions['ce'].get('symbol'),
                'ce_entry_price': self.positions['ce'].get('entry_price'),
                'ce_ltp': self.positions['ce'].get('ltp'),
                'ce_sl_price': self.positions['ce'].get('current_sl'),
                'ce_real_entered': self.positions['ce'].get('real_entered'),
                # main PE
                'pe_symbol': self.positions['pe'].get('symbol'),
                'pe_entry_price': self.positions['pe'].get('entry_price'),
                'pe_ltp': self.positions['pe'].get('ltp'),
                'pe_sl_price': self.positions['pe'].get('current_sl'),
                'pe_real_entered': self.positions['pe'].get('real_entered'),
                # hedges
                'ce_hedge_symbol': self.hedges['ce'].get('symbol'),
                'ce_hedge_entry': self.hedges['ce'].get('entry_price'),
                'pe_hedge_symbol': self.hedges['pe'].get('symbol'),
                'pe_hedge_entry': self.hedges['pe'].get('entry_price'),
            }
            row.update(extra)
            df = pd.DataFrame([row])
            write_header = not os.path.exists(self.current_state_file)
            df.to_csv(self.current_state_file, mode='a', header=write_header, index=False)
            logger.debug("State logged: %s", status)
        except Exception as e:
            logger.error("_log_state failed: %s\n%s", e, traceback.format_exc())

    # Convenience wrapper for logging short rows
    def _log(self, status: str, comments: str = ""):
        self._log_state(status, comments)

