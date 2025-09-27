"""
intraday_straddle_strategy.py

Intraday Straddle strategy (enhanced):
 - Straddle selection: finds CE+PE at SAME strike using 50-step offsets around spot.
   Checks in order: round down/up to 50, then ±50, ±100, ±150 ... until a best candidate is found.
   Chooses strike whose |CE_ltp - PE_ltp| is smallest among candidates (both LTPs available).
 - Virtual CE+PE created at 09:20 IST.
 - When any virtual leg's virtual SL is hit, take a real SELL on the OPPOSITE leg (same strike).
 - INITIAL_SL_MULTIPLIER = 1.20 (20%).
 - When activating the real trade, also place a hedge BUY (opposite side) on a far OTM option whose premium is in HEDGE_RANGE.
 - State file & logs prefixed with intraday_straddle_
"""

import os
import sys
import logging
import math
import traceback
from datetime import datetime, time, timedelta
from functools import wraps
from typing import Optional, Dict, Any, Tuple, List, Union
from PyQt5.QtCore import QTimer
import pandas as pd
from pytz import timezone

# ----------------------------- CONFIGURATION / CONSTANTS -----------------------------
IST = timezone('Asia/Kolkata')
LOGGER_NAME = __name__

# Strategy identity
STRATEGY_NAME = "intraday_straddle"

# Trading windows (IST)
TRADING_START_TIME = time(9, 20)
TRADING_END_TIME = time(15, 15)
VIRTUAL_ENTRY_MINUTE = 20

# Strike step (user requested 50-step)
STRIKE_STEP = 50

# Stop-loss & trailing SL
INITIAL_SL_MULTIPLIER = 1.20   # 20% as requested
SL_ROUNDING_FACTOR = 20
TRAILING_SL_STEPS = [0.10, 0.20, 0.30, 0.40, 0.50]

# Premium selection ranges
PREMIUM_RANGE = (1.0, 9999.0)     # for evaluating straddle candidates we accept any positive premium; actual selection is by diff
HEDGE_RANGE = (5.0, 15.0)         # user requested 5-15 price for hedge

# How far to search for candidate strikes (steps of 50)
MAX_STRIKE_DISTANCE = 500   # i.e., search up to ±500 from rounded strikes (change if you want deeper search)

# Timers (ms)
STRATEGY_CHECK_INTERVAL = 60_000
MONITORING_INTERVAL = 10_000

# Files & directories (prefixed)
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
STATE_FILE_TEMPLATE = os.path.join(LOG_DIR, "intraday_straddle_{date}_state.csv")

# Order config
LOT_SIZE = 75
ORDER_PRODUCT_TYPE = 'M'
ORDER_EXCHANGE = 'NFO'

# NFO symbols file (expects columns like Instrument, Symbol (underlying), Expiry, Strike, OptionType, TradingSymbol, Token)
NFO_SYMBOLS_FILE = "NFO_symbols.txt"

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(LOGGER_NAME)

# ----------------------------- UTILITIES --------------------------------
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
    def current_time() -> time:
        return ISTTimeUtils.now().time()

    @staticmethod
    def current_date_str() -> str:
        return ISTTimeUtils.now().strftime("%Y-%m-%d")

def round_sl(price: float) -> float:
    """Round SL to nearest tick defined by SL_ROUNDING_FACTOR."""
    return math.ceil(price * SL_ROUNDING_FACTOR) / SL_ROUNDING_FACTOR

def nearest_50_floor(spot: float) -> int:
    return int(math.floor(spot / STRIKE_STEP) * STRIKE_STEP)

def nearest_50_ceil(spot: float) -> int:
    return int(math.ceil(spot / STRIKE_STEP) * STRIKE_STEP)

# ----------------------------- STRATEGY CLASS ------------------------------------
class IntradayStraddleStrategy:
    def __init__(self, ui: Optional[Any] = None, client_manager: Optional[Any] = None):
        self.ui = ui
        self.client_manager = client_manager

        self.intraday_straddle_state_file = STATE_FILE_TEMPLATE.format(date=ISTTimeUtils.current_date_str())
        self.intraday_straddle_state: str = "WAITING"
        self.intraday_straddle_positions = {
            'ce': self._empty_position(),
            'pe': self._empty_position()
        }
        self.intraday_straddle_hedge_info: Dict[str, Any] = {}

        # timers
        self.strategy_timer = QTimer()
        try:
            self.strategy_timer.timeout.connect(self._check_and_execute_strategy)
            self.strategy_timer.start(STRATEGY_CHECK_INTERVAL)
        except Exception:
            logger.debug("QTimer unavailable to start strategy_timer.")

        self.monitor_timer = QTimer()
        try:
            self.monitor_timer.timeout.connect(self._monitor_all)
            self.monitor_timer.start(MONITORING_INTERVAL)
        except Exception:
            logger.debug("QTimer unavailable to start monitor_timer.")

        # recover
        self._try_recover_from_state_file()

    def _empty_position(self) -> Dict[str, Any]:
        return {
            'symbol': None, 'token': None, 'strike': None, 'ltp': 0.0,
            'entry_price': 0.0, 'initial_sl': 0.0, 'current_sl': 0.0,
            'sl_hit': False, 'max_profit_price': 0.0, 'trailing_step': 0,
            'real_entered': False, 'virtual': True
        }

    @safe_log("recovery")
    def _try_recover_from_state_file(self):
        if not os.path.exists(self.intraday_straddle_state_file):
            logger.info("No intraday_straddle state file to recover.")
            return
        try:
            df = pd.read_csv(self.intraday_straddle_state_file, on_bad_lines='skip')
            if df.empty:
                return
            last = df.iloc[-1]
            self.intraday_straddle_state = str(last.get('status', 'WAITING'))
            for leg in ['ce','pe']:
                sym = last.get(f'{leg}_symbol', None)
                if pd.notna(sym) and str(sym).strip():
                    self.intraday_straddle_positions[leg]['symbol'] = str(sym).strip()
                    self.intraday_straddle_positions[leg]['entry_price'] = float(last.get(f'{leg}_entry_price', 0.0))
                    self.intraday_straddle_positions[leg]['ltp'] = float(last.get(f'{leg}_ltp', 0.0))
                    self.intraday_straddle_positions[leg]['initial_sl'] = float(last.get(f'{leg}_sl_price', 0.0))
                    self.intraday_straddle_positions[leg]['current_sl'] = float(last.get(f'{leg}_sl_price', 0.0))
                    self.intraday_straddle_positions[leg]['real_entered'] = bool(last.get(f'{leg}_real_entered', False))
                    self.intraday_straddle_positions[leg]['strike'] = last.get(f'{leg}_strike', None)
            logger.info("Recovered intraday_straddle_state=%s", self.intraday_straddle_state)
        except Exception as e:
            logger.warning("Recovery reading failed: %s", e)

    # -------------------- Entry cycle --------------------
    def _check_and_execute_strategy(self):
        now = ISTTimeUtils.current_time()
        # EOD forced exit
        if (now.hour == TRADING_END_TIME.hour and now.minute == TRADING_END_TIME.minute) and self.intraday_straddle_state in ['VIRTUAL', 'ACTIVE']:
            self._exit_all_positions(reason="EOD Exit")
            return

        if self.intraday_straddle_state == "WAITING" and TRADING_START_TIME <= now <= TRADING_END_TIME and now.minute == VIRTUAL_ENTRY_MINUTE:
            logger.info("Virtual entry minute hit -> prepare virtual straddle")
            self._run_virtual_entry()

    def _run_virtual_entry(self):
        expiry_date = self._get_current_expiry()
        if not expiry_date:
            logger.error("Expiry not found. Cannot run virtual entry.")
            return
        # get spot price (NIFTY underlying)
        spot = self._get_spot_ltp()
        if spot is None:
            logger.error("Could not fetch spot LTP (NIFTY). Aborting virtual entry.")
            return

        best = self._find_best_straddle(expiry_date, spot)
        if not best:
            logger.info("No suitable straddle found at virtual entry minute.")
            return

        strike, ce_sym, ce_token, ce_ltp, pe_sym, pe_token, pe_ltp = best
        # set both legs as virtual shorts
        initial_sl_ce = round_sl(ce_ltp * INITIAL_SL_MULTIPLIER)
        initial_sl_pe = round_sl(pe_ltp * INITIAL_SL_MULTIPLIER)
        self.intraday_straddle_positions['ce'].update({
            'symbol': ce_sym, 'token': ce_token, 'strike': strike,
            'entry_price': ce_ltp, 'ltp': ce_ltp, 'initial_sl': initial_sl_ce,
            'current_sl': initial_sl_ce, 'real_entered': False, 'virtual': True
        })
        self.intraday_straddle_positions['pe'].update({
            'symbol': pe_sym, 'token': pe_token, 'strike': strike,
            'entry_price': pe_ltp, 'ltp': pe_ltp, 'initial_sl': initial_sl_pe,
            'current_sl': initial_sl_pe, 'real_entered': False, 'virtual': True
        })
        self.intraday_straddle_state = "VIRTUAL"
        self._log_state("VIRTUAL", f"Virtual straddle created at strike {strike}")
        logger.info("Virtual straddle (strike=%s) prepared: CE=%s(%s) PE=%s(%s)", strike, ce_sym, ce_ltp, pe_sym, pe_ltp)

    def _find_best_straddle(self, expiry_date, spot: float):
        """
        Search strike candidates around spot with 50-step increments as user requested:
        order: floor/ceil, then ±50, ±100, ±150...
        For each candidate, get CE and PE LTPs (from NFO_symbols file & broker quotes).
        Choose the candidate where both LTPs exist and |CE_ltp - PE_ltp| is smallest.
        """
        try:
            if not os.path.exists(NFO_SYMBOLS_FILE):
                logger.error("%s missing", NFO_SYMBOLS_FILE)
                return None
            df = pd.read_csv(NFO_SYMBOLS_FILE)
            expiry_str = expiry_date.strftime("%d-%b-%Y").upper()

            floor = nearest_50_floor(spot)
            ceil = nearest_50_ceil(spot)
            # generate candidate strikes in requested order:
            candidates = []
            # start with floor then ceil
            candidates.append(floor)
            if ceil != floor:
                candidates.append(ceil)
            # then expand +/-50, +/-100...
            steps = list(range(STRIKE_STEP, MAX_STRIKE_DISTANCE + STRIKE_STEP, STRIKE_STEP))
            for s in steps:
                lower = floor - s
                upper = ceil + s
                if lower > 0:
                    candidates.append(lower)
                candidates.append(upper)

            best = None
            best_diff = float('inf')
            for strike in candidates:
                # find CE and PE rows
                ce_row = df[
                    (df['Instrument'].str.strip() == "OPTIDX") &
                    (df['Symbol'].str.strip() == "NIFTY") &
                    (df['Expiry'].str.strip().str.upper() == expiry_str) &
                    (df['OptionType'].str.strip().str.upper() == "CE") &
                    (pd.to_numeric(df['Strike'], errors='coerce') == strike)
                ]
                pe_row = df[
                    (df['Instrument'].str.strip() == "OPTIDX") &
                    (df['Symbol'].str.strip() == "NIFTY") &
                    (df['Expiry'].str.strip().str.upper() == expiry_str) &
                    (df['OptionType'].str.strip().str.upper() == "PE") &
                    (pd.to_numeric(df['Strike'], errors='coerce') == strike)
                ]
                if ce_row.empty or pe_row.empty:
                    continue
                # take first candidates
                ce_row = ce_row.iloc[0]
                pe_row = pe_row.iloc[0]
                ce_sym = str(ce_row.get('TradingSymbol','')).strip()
                ce_token = str(ce_row.get('Token','')).strip()
                pe_sym = str(pe_row.get('TradingSymbol','')).strip()
                pe_token = str(pe_row.get('Token','')).strip()
                ce_ltp = self._get_option_ltp(ce_sym)
                pe_ltp = self._get_option_ltp(pe_sym)
                if ce_ltp is None or pe_ltp is None:
                    continue
                diff = abs(ce_ltp - pe_ltp)
                # prefer smaller diff
                if diff < best_diff:
                    best_diff = diff
                    best = (strike, ce_sym, ce_token, ce_ltp, pe_sym, pe_token, pe_ltp)
                    # if diff is small enough (like <=5) we can break early; but to be faithful, continue search to find best min
            return best
        except Exception as e:
            logger.error("_find_best_straddle error: %s\n%s", e, traceback.format_exc())
            return None

    # -------------------- Monitoring --------------------
    def _monitor_all(self):
        now = ISTTimeUtils.current_time()
        if not (TRADING_START_TIME <= now <= TRADING_END_TIME):
            return

        if self.intraday_straddle_state == "VIRTUAL":
            self._monitor_virtual_sl()
        elif self.intraday_straddle_state == "ACTIVE":
            self._monitor_active_positions()
            self._validate_active_positions_exist()

    def _monitor_virtual_sl(self):
        """
        If any virtual leg LTP >= virtual SL, activate the OPPOSITE real SELL on same strike
        and also place the hedge BUY (opposite side far OTM with premium in HEDGE_RANGE).
        """
        try:
            for leg in ['ce','pe']:
                pos = self.intraday_straddle_positions[leg]
                if not pos['symbol']:
                    continue
                ltp = self._get_option_ltp(pos['symbol'])
                if ltp is None:
                    continue
                pos['ltp'] = ltp
                if not pos.get('sl_hit', False) and ltp >= pos.get('current_sl', float('inf')):
                    logger.info("Virtual %s SL hit (LTP=%s >= SL=%s). Activating real SELL on opposite leg.", leg.upper(), ltp, pos['current_sl'])
                    opposite_leg = 'pe' if leg == 'ce' else 'ce'
                    self._activate_real_on_opposite_and_hedge(opposite_leg)
                    return
        except Exception as e:
            logger.error("_monitor_virtual_sl error: %s\n%s", e, traceback.format_exc())

    def _activate_real_on_opposite_and_hedge(self, opp_leg: str) -> bool:
        """
        Place hedge BUY first (mandatory).
        Only if hedge BUY succeeds, place real SELL on opp_leg (same strike).
        If main SELL fails, auto-exit the hedge to avoid margin / exposure issues.
        """
        try:
            if opp_leg not in ['ce','pe']:
                logger.error("Invalid activation leg: %s", opp_leg)
                return False

            pos = self.intraday_straddle_positions[opp_leg]
            if not pos['symbol']:
                logger.error("Opposite leg symbol missing for activation.")
                return False

            # --- Step 1: find hedge ---
            hedge = self._find_hedge_option(opp_leg, pos['strike'])
            if not hedge:
                logger.error("No hedge candidate found in HEDGE_RANGE for %s. Aborting activation.", opp_leg.upper())
                return False

            hedge_sym, hedge_token, hedge_ltp = hedge
            hedge_ok = self._place_order(hedge_sym, hedge_token, 'BUY')
            if not hedge_ok:
                logger.error("Hedge BUY failed for %s. Aborting activation.", hedge_sym)
                return False

            # record hedge info
            self.intraday_straddle_hedge_info = {
                'hedge_symbol': hedge_sym,
                'hedge_token': hedge_token,
                'hedge_ltp': hedge_ltp,
                'hedge_for': opp_leg
            }
            logger.info("Hedge placed successfully: BUY %s @ %s", hedge_sym, hedge_ltp)

            # --- Step 2: place main real SELL ---
            sell_ok = self._place_order(pos['symbol'], pos['token'], 'SELL')
            if not sell_ok:
                logger.error("Main SELL failed for %s after hedge BUY. Exiting hedge to stay clean.", pos['symbol'])
                # try to exit hedge immediately
                try:
                    self._place_order(hedge_sym, hedge_token, 'SELL')
                    logger.info("Exited hedge %s after main SELL failure.", hedge_sym)
                except Exception as e:
                    logger.error("Failed to exit hedge %s after main SELL failure: %s", hedge_sym, e)
                return False

            # update position info
            actual_ltp = self._get_option_ltp(pos['symbol']) or pos['entry_price']
            new_sl = round_sl(actual_ltp * INITIAL_SL_MULTIPLIER)
            pos.update({
                'entry_price': actual_ltp,
                'ltp': actual_ltp,
                'initial_sl': new_sl,
                'current_sl': new_sl,
                'real_entered': True,
                'virtual': False,
                'max_profit_price': actual_ltp,
                'trailing_step': 0,
                'sl_hit': False
            })

            self.intraday_straddle_state = "ACTIVE"
            self._log_state("ACTIVE", f"Real SELL placed on {opp_leg.upper()} after hedge {hedge_sym}")
            logger.info("Activation complete: Hedge BUY=%s, Main SELL=%s", hedge_sym, pos['symbol'])
            return True

        except Exception as e:
            logger.error("_activate_real_on_opposite_and_hedge error: %s\n%s", e, traceback.format_exc())
            return False



    def _find_hedge_option(self, real_leg: str, base_strike: int):
        """
        For a real SELL on real_leg, find a hedge option to BUY on the opposite side with premium in HEDGE_RANGE.
        - If real_leg == 'ce' (we sold CE), hedge by BUYing PE at lower strikes (OTM PE -> strike < base_strike)
        - If real_leg == 'pe' (we sold PE), hedge by BUYing CE at higher strikes (OTM CE -> strike > base_strike)
        Search outward in 50 increments and pick the first option whose LTP in HEDGE_RANGE.
        """
        try:
            if not os.path.exists(NFO_SYMBOLS_FILE):
                return None
            df = pd.read_csv(NFO_SYMBOLS_FILE)
            expiry_date = self._get_current_expiry()
            if not expiry_date:
                return None
            expiry_str = expiry_date.strftime("%d-%b-%Y").upper()

            # determine desired option side for hedge
            if real_leg == 'ce':
                hedge_side = 'PE'
                # search lower strikes: base -50, -100, ...
                offsets = [i for i in range(STRIKE_STEP, MAX_STRIKE_DISTANCE + STRIKE_STEP, STRIKE_STEP)]
                strikes = [base_strike - off for off in offsets if base_strike - off > 0]
            else:
                hedge_side = 'CE'
                offsets = [i for i in range(STRIKE_STEP, MAX_STRIKE_DISTANCE + STRIKE_STEP, STRIKE_STEP)]
                strikes = [base_strike + off for off in offsets]

            for strike in strikes:
                candidates = df[
                    (df['Instrument'].str.strip() == "OPTIDX") &
                    (df['Symbol'].str.strip() == "NIFTY") &
                    (df['Expiry'].str.strip().str.upper() == expiry_str) &
                    (df['OptionType'].str.strip().str.upper() == hedge_side) &
                    (pd.to_numeric(df['Strike'], errors='coerce') == strike)
                ]
                if candidates.empty:
                    continue
                row = candidates.iloc[0]
                sym = str(row.get('TradingSymbol','')).strip()
                token = str(row.get('Token','')).strip()
                ltp = self._get_option_ltp(sym)
                if ltp is None:
                    continue
                # accept if ltp in HEDGE_RANGE
                if HEDGE_RANGE[0] <= ltp <= HEDGE_RANGE[1]:
                    return sym, token, ltp
            return None
        except Exception as e:
            logger.error("_find_hedge_option error: %s\n%s", e, traceback.format_exc())
            return None

    # -------------------- Active monitoring & exit --------------------
    def _monitor_active_positions(self):
        try:
            for leg in ['ce','pe']:
                pos = self.intraday_straddle_positions[leg]
                if not pos['symbol'] or not pos.get('real_entered', False):
                    continue
                ltp = self._get_option_ltp(pos['symbol'])
                if ltp is None:
                    continue
                pos['ltp'] = ltp
                # update max_profit_price for short (profit when ltp drops)
                if pos.get('max_profit_price', 0.0) == 0.0:
                    pos['max_profit_price'] = pos['entry_price']
                if ltp < pos.get('max_profit_price', pos['entry_price']):
                    pos['max_profit_price'] = ltp
                self._update_trailing_sl(leg)
                current_sl = pos.get('current_sl', None)
                if current_sl and not pos.get('sl_hit', False) and ltp >= current_sl:
                    logger.info("Real %s SL hit: LTP=%s >= SL=%s — exiting real SELL", leg.upper(), ltp, current_sl)
                    self._exit_real_position(leg, ltp)
        except Exception as e:
            logger.error("_monitor_active_positions error: %s\n%s", e, traceback.format_exc())

    def _update_trailing_sl(self, leg: str):
        try:
            pos = self.intraday_straddle_positions[leg]
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
                self._log_state("ACTIVE", f"{leg.upper()} trailing SL updated to {new_sl}", trailing_step=new_step)
        except Exception as e:
            logger.error("_update_trailing_sl error: %s\n%s", e, traceback.format_exc())

    def _exit_real_position(self, leg: str, ltp: float):
        try:
            pos = self.intraday_straddle_positions[leg]
            if not pos['symbol'] or not pos.get('real_entered', False):
                logger.warning("No real position to exit for %s", leg)
                return False
            ok = self._place_order(pos['symbol'], pos['token'], 'BUY')
            if ok:
                pos['sl_hit'] = True
                pos['real_entered'] = False
                self._log_state("STOPPED_OUT", f"{leg.upper()} SL hit at {ltp}", exit_ltp=ltp)
                # attempt to close hedge if exists (we bought hedge earlier; we will SELL to close)
                hedge = self.intraday_straddle_hedge_info.get('hedge_symbol')
                if hedge:
                    try:
                        self._place_order(hedge, self.intraday_straddle_hedge_info.get('hedge_token',''), 'SELL')
                        logger.info("Closed hedge %s (intraday_straddle).", hedge)
                    except Exception:
                        logger.warning("Failed to close hedge %s", hedge)
                # if both real legs closed -> STOPPED_OUT
                both_closed = all(not self.intraday_straddle_positions[l]['real_entered'] for l in ['ce','pe'])
                if both_closed:
                    self.intraday_straddle_state = "STOPPED_OUT"
                return True
            else:
                logger.error("Failed to exit real position %s via BUY", pos['symbol'])
                return False
        except Exception as e:
            logger.error("_exit_real_position error: %s\n%s", e, traceback.format_exc())
            return False

    def _exit_all_positions(self, reason: str = ""):
        logger.info("intraday_straddle: exiting all positions - %s", reason)
        all_closed_successfully = True
        # close mains (BUY) if real_entered
        for leg in ['ce','pe']:
            pos = self.intraday_straddle_positions[leg]
            if pos['symbol'] and pos.get('real_entered', False):
                try:
                    ok = self._place_order(pos['symbol'], pos['token'], 'BUY')
                    if ok:
                        self.intraday_straddle_positions[leg] = self._empty_position()
                    else:
                        all_closed_successfully = False
                except Exception:
                    all_closed_successfully = False
        # close hedge if present (SELL hedge)
        hedge_sym = self.intraday_straddle_hedge_info.get('hedge_symbol')
        if hedge_sym:
            try:
                self._place_order(hedge_sym, self.intraday_straddle_hedge_info.get('hedge_token',''), 'SELL')
            except Exception:
                pass

        if all_closed_successfully:
            self.intraday_straddle_state = "COMPLETED"
            self._log_state("COMPLETED", f"All closed: {reason}")
        else:
            self._log_state(self.intraday_straddle_state, f"Exit attempted: {reason}")

    # -------------------- Broker / Quote / Order wrappers --------------------
    def _get_spot_ltp(self) -> Optional[float]:
        """
        Fetch spot LTP for NIFTY (underlying). We expect NIFTY index symbol available in client quotes (like NIFTY or NIFTY 50).
        If your broker uses a specific underlying trading symbol, adapt this function.
        """
        try:
            client = self._get_primary_client()
            if not client:
                return None
            # Try common NIFTY identifiers - adapt if different in your client
            for ident in ['NIFTY 50', 'NIFTY', 'NIFTYII', 'NIFTY50']:
                try:
                    q = client.get_quotes('NSE', ident)
                    if q and isinstance(q, dict):
                        if 'lp' in q and q.get('lp') is not None:
                            return float(q.get('lp'))
                        if 'data' in q and isinstance(q['data'], dict):
                            for v in q['data'].values():
                                if isinstance(v, dict) and 'lp' in v and v.get('lp') is not None:
                                    return float(v.get('lp'))
                except Exception:
                    continue
            # fallback: try underlying from NFO_symbols file—look for Instrument 'INDEX' and Symbol 'NIFTY'
            try:
                if os.path.exists(NFO_SYMBOLS_FILE):
                    df = pd.read_csv(NFO_SYMBOLS_FILE)
                    idx_row = df[(df['Instrument'].str.strip() == 'INDEX') & (df['Symbol'].str.strip() == 'NIFTY')]
                    if not idx_row.empty:
                        sym = idx_row.iloc[0].get('TradingSymbol','')
                        q = client.get_quotes('NSE', sym)
                        if q and isinstance(q, dict):
                            if 'lp' in q and q.get('lp') is not None:
                                return float(q.get('lp'))
            except Exception:
                pass
            return None
        except Exception as e:
            logger.debug("_get_spot_ltp error: %s", e)
            return None

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
            logger.debug("_get_option_ltp error: %s", e)
            return None

    def _place_order(self, symbol: str, token: str, action: str) -> bool:
        try:
            client = self._get_primary_client()
            if not client:
                logger.error("No trading client available")
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
            # best-effort success check
            if isinstance(order_res, dict):
                if order_res.get('stat') in ['Ok','OK','ok'] or order_res.get('status','').lower() in ['success','ok']:
                    return True
            if isinstance(order_res, bool):
                return order_res
            return True
        except Exception as e:
            logger.error("_place_order failed: %s\n%s", e, traceback.format_exc())
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
            logger.warning("Failed to get broker positions: %s", e)
            return []

    def _get_primary_client(self):
        try:
            if self.client_manager and hasattr(self.client_manager, "clients") and self.client_manager.clients:
                return self.client_manager.clients[0][2]
        except Exception:
            logger.exception("Error fetching primary client")
        return None

    def _is_symbol_in_pos(self, symbol: Optional[str], broker_pos: Dict[str, Any]) -> bool:
        if not symbol or not broker_pos:
            return False
        try:
            norm = str(symbol).strip().upper()
            for key in ['tradingsymbol', 'TradingSymbol', 'symbol', 'Trading_Symbol', 'instrument']:
                if key in broker_pos and str(broker_pos[key]).strip().upper() == norm:
                    return True
        except Exception:
            return False
        return False

    def _validate_active_positions_exist(self):
        try:
            if self.intraday_straddle_state != "ACTIVE":
                return
            broker_positions = self._get_broker_positions()
            found_ce = any(self._is_symbol_in_pos(self.intraday_straddle_positions['ce']['symbol'], bp) for bp in broker_positions) if self.intraday_straddle_positions['ce']['symbol'] else False
            found_pe = any(self._is_symbol_in_pos(self.intraday_straddle_positions['pe']['symbol'], bp) for bp in broker_positions) if self.intraday_straddle_positions['pe']['symbol'] else False
            if not found_ce and not found_pe:
                logger.info("ACTIVE but no broker positions detected for CE/PE.")
                # reset to WAITING for safety
                self.intraday_straddle_positions = {'ce': self._empty_position(), 'pe': self._empty_position()}
                self.intraday_straddle_state = "WAITING"
                self._log_state("WAITING", "Auto-reset: no broker positions detected (intraday_straddle)")
        except Exception as e:
            logger.error("_validate_active_positions_exist: %s\n%s", e, traceback.format_exc())

    def _get_current_expiry(self):
        """
        Uses UI dropdown if present; otherwise returns None.
        """
        try:
            if not self.ui or not hasattr(self.ui, 'ExpiryListDropDown'):
                return None
            expiry_dates = []
            for i in range(self.ui.ExpiryListDropDown.count()):
                txt = self.ui.ExpiryListDropDown.itemText(i)
                try:
                    d = datetime.strptime(txt, "%d-%b-%Y").date()
                    expiry_dates.append(d)
                except Exception:
                    continue
            expiry_dates.sort()
            today = datetime.now().date()
            for d in expiry_dates:
                if d >= today:
                    return d
            return expiry_dates[-1] if expiry_dates else None
        except Exception:
            return None

    # -------------------- State logging --------------------
    def _log_state(self, status: str, comments: str = "", **extra):
        try:
            now = ISTTimeUtils.now().strftime("%Y-%m-%d %H:%M:%S")
            row = {
                'timestamp': now,
                'status': status,
                'comments': comments,
                'ce_symbol': self.intraday_straddle_positions['ce'].get('symbol'),
                'ce_strike': self.intraday_straddle_positions['ce'].get('strike'),
                'ce_entry_price': self.intraday_straddle_positions['ce'].get('entry_price'),
                'ce_ltp': self.intraday_straddle_positions['ce'].get('ltp'),
                'ce_sl_price': self.intraday_straddle_positions['ce'].get('current_sl'),
                'ce_real_entered': self.intraday_straddle_positions['ce'].get('real_entered'),
                'pe_symbol': self.intraday_straddle_positions['pe'].get('symbol'),
                'pe_strike': self.intraday_straddle_positions['pe'].get('strike'),
                'pe_entry_price': self.intraday_straddle_positions['pe'].get('entry_price'),
                'pe_ltp': self.intraday_straddle_positions['pe'].get('ltp'),
                'pe_sl_price': self.intraday_straddle_positions['pe'].get('current_sl'),
                'pe_real_entered': self.intraday_straddle_positions['pe'].get('real_entered'),
                'hedge_symbol': self.intraday_straddle_hedge_info.get('hedge_symbol'),
                'hedge_ltp': self.intraday_straddle_hedge_info.get('hedge_ltp')
            }
            row.update(extra)
            df = pd.DataFrame([row])
            write_header = not os.path.exists(self.intraday_straddle_state_file)
            df.to_csv(self.intraday_straddle_state_file, mode='a', header=write_header, index=False)
        except Exception as e:
            logger.error("_log_state failed: %s\n%s", e, traceback.format_exc())

