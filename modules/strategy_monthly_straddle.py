# strategy_monthly_straddle.py
import os
import logging
import math
import pandas as pd
from datetime import datetime, date, time, timedelta
from functools import wraps
from PyQt5.QtCore import QTimer
from pytz import timezone
from calendar import monthrange
from PyQt5.QtWidgets import QMessageBox
from typing import Optional, Dict, Any, List

# ===== CONFIGURATION =====
IST = timezone('Asia/Kolkata')
LOT_SIZE = 75
LOG_DIR_NAME = "logs"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    os.makedirs(LOG_DIR_NAME, exist_ok=True)
    fh = logging.FileHandler(os.path.join(LOG_DIR_NAME, "monthly_straddle_debug.log"))
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

# ===== UTILITIES =====
def safe_log(context: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(f"{context}: {e}")
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

    @staticmethod
    def is_trading_day() -> bool:
        return ISTTimeUtils.now().weekday() < 5

# ===== STRATEGY CLASS =====
class MonthlyStraddleStrategy:
    def __init__(self, ui, client_manager, position_manager=None, nfo_symbols_path="NFO_symbols.txt"):
        logger.info("Initializing MonthlyStraddleStrategy")
        self.ui = ui
        self.client_manager = client_manager
        self.position_manager = position_manager
        self.nfo_symbols_path = nfo_symbols_path

        # Strategy state
        self.spot_price = None
        self.entry_spot_price = None
        self.expiry_date = None
        self.entry_date = None
        self.total_premium_received = 0.0
        self.premium_target = 0.0
        self.lower_range = None
        self.higher_range = None
        self.max_premium = 0.0

        # Positions and state
        self.state = "WAITING"
        self.positions = {
            "ce_sell": None,
            "pe_sell": None,
            "ce_buy": None,
            "pe_buy": None
        }

        # Files
        self.log_dir = os.path.join(os.path.dirname(__file__), LOG_DIR_NAME)
        os.makedirs(self.log_dir, exist_ok=True)
        self.current_state_file = os.path.join(self.log_dir, f"{ISTTimeUtils.current_date_str()}_monthly_straddle_state.csv")
        self.analysis_file = os.path.join(self.log_dir, "monthly_straddle_analysis.csv")

        # Timer
        self.strategy_timer = QTimer()
        self.strategy_timer.timeout.connect(self._check_and_execute_strategy)
        self.strategy_timer.start(30 * 1000)

        logger.info("Strategy initialized and timer started (30s interval)")

        # UI integration
        if hasattr(self.ui, "ExecuteMonthlyStraddleQPushButton"):
            try:
                self.ui.ExecuteMonthlyStraddleQPushButton.clicked.connect(self.on_execute_strategy_clicked)
            except Exception:
                logger.debug("UI button hookup failed or not present")

    # ===== Public Methods =====
    def attempt_recovery(self):
        logger.info("Attempting Monthly Straddle strategy recovery")
        if self.recover_from_state_file():
            logger.info("Monthly Straddle recovered from state file")
            return True
        elif self.recover_from_positions():
            logger.info("Monthly Straddle recovered from positions")
            return True
        else:
            logger.info("Monthly Straddle recovery failed")
            return False

    def recover_from_state_file(self):
        try:
            state_file_path = self._find_latest_state_file()
            if not state_file_path:
                logger.info("No state file found for recovery")
                return False

            df = pd.read_csv(state_file_path)
            if df.empty:
                return False
                
            latest_state = df.iloc[-1]

            if latest_state.get('status') != 'ACTIVE':
                logger.info(f"Last state was {latest_state.get('status')}, not recovering")
                return False

            self.state = latest_state.get('status', 'WAITING')
            self.entry_spot_price = float(latest_state.get('entry_spot_price', 0.0))
            self.total_premium_received = float(latest_state.get('total_premium', 0.0))
            self.premium_target = float(latest_state.get('premium_target', 0.0))
            self.lower_range = float(latest_state.get('lower_range', 0.0))
            self.higher_range = float(latest_state.get('higher_range', 0.0))
            self.max_premium = float(latest_state.get('max_premium', 0.0))

            # Parse dates
            self._parse_dates_from_state(latest_state)

            # Recover positions
            position_legs = ['ce_sell', 'pe_sell', 'ce_buy', 'pe_buy']
            for leg in position_legs:
                symbol_col = f"{leg}_symbol"
                token_col = f"{leg}_token"
                entry_col = f"{leg}_entry"
                qty_col = f"{leg}_qty"
                strike_col = f"{leg}_strike"
                option_type_col = f"{leg}_option_type"

                symbol = latest_state.get(symbol_col)
                if pd.notna(symbol) and symbol:
                    self.positions[leg] = {
                        'symbol': symbol,
                        'token': str(latest_state.get(token_col, '')),
                        'entry_price': float(latest_state.get(entry_col, 0.0)),
                        'net_qty': int(latest_state.get(qty_col, 0)),
                        'ltp': float(latest_state.get(entry_col, 0.0)),
                        'strike': latest_state.get(strike_col),
                        'option_type': latest_state.get(option_type_col)
                    }

            logger.info(f"Strategy recovered from state file: {self.state}")
            logger.info(f"Recovered {sum(1 for p in self.positions.values() if p)} positions")

            # Update current market data
            self._get_spot_price()

            # Register with PositionManager
            self.register_strategy_positions()

            self._log_state(
                "ACTIVE",
                "Recovered from state file",
                expiry_date=self.expiry_date,
                entry_date=self.entry_date
            )

            return True

        except Exception as e:
            logger.error(f"State file recovery failed: {e}")
            return False

    def recover_from_positions(self, positions_list=None):
        try:
            logger.info(f"Recovering Monthly Straddle from positions")
            
            if positions_list is None:
                positions_list = self._get_strategy_positions_from_broker()

            if not positions_list:
                logger.info("No positions found for recovery")
                return False

            # Separate short positions (core) from long positions (hedges)
            short_positions = []
            long_positions = []
            
            for pos in positions_list:
                symbol = pos.get('symbol', '')
                avg_price = pos.get('avg_price', 0)
                net_qty = pos.get('net_qty', 0)
                entry_spot_price = pos.get('entry_spot_price', 0)
                
                # Determine option type
                if 'CE' in symbol or 'C' in symbol or 'C' in symbol[-6:]:
                    option_type = 'CE'
                elif 'PE' in symbol or 'P' in symbol or 'P' in symbol[-6:]:
                    option_type = 'PE'
                else:
                    option_type = None
                    logger.warning(f"Could not determine option type for symbol: {symbol}")

                if net_qty < 0:  # Short positions
                    short_positions.append({
                        'symbol': symbol, 
                        'avg_price': avg_price, 
                        'net_qty': net_qty,
                        'option_type': option_type
                    })
                    # Use the first valid spot price found
                    if entry_spot_price and self.entry_spot_price is None:
                        self.entry_spot_price = entry_spot_price
                        
                elif net_qty > 0:  # Long positions (hedges)
                    long_positions.append({
                        'symbol': symbol,
                        'avg_price': avg_price,
                        'net_qty': net_qty, 
                        'option_type': option_type
                    })

            logger.info(f"Short positions: {len(short_positions)}, Long positions: {len(long_positions)}")
            logger.info(f"Entry spot price: {self.entry_spot_price}")

            # Check if we have the core short straddle
            short_ce = next((p for p in short_positions if p['option_type'] == 'CE'), None)
            short_pe = next((p for p in short_positions if p['option_type'] == 'PE'), None)

            if not short_ce or not short_pe:
                logger.warning("Missing core short straddle positions")
                return False

            # If spot price is 0, try to get it from CSV or estimate
            if self.entry_spot_price == 0.0:
                self.entry_spot_price = self._get_spot_price_from_csv() or 25000
                logger.info(f"Using spot price: {self.entry_spot_price}")

            # Recover core short positions
            self.state = "ACTIVE"
            self.positions["ce_sell"] = {
                'symbol': short_ce['symbol'],
                'entry_price': short_ce['avg_price'],
                'net_qty': short_ce['net_qty'],
                'ltp': short_ce['avg_price']
            }
            self.positions["pe_sell"] = {
                'symbol': short_pe['symbol'], 
                'entry_price': short_pe['avg_price'],
                'net_qty': short_pe['net_qty'],
                'ltp': short_pe['avg_price']
            }

            # Recover hedge positions if available
            for hedge in long_positions:
                if hedge['option_type'] == 'CE':
                    self.positions["ce_buy"] = {
                        'symbol': hedge['symbol'],
                        'entry_price': hedge['avg_price'], 
                        'net_qty': hedge['net_qty']
                    }
                elif hedge['option_type'] == 'PE':
                    self.positions["pe_buy"] = {
                        'symbol': hedge['symbol'],
                        'entry_price': hedge['avg_price'],
                        'net_qty': hedge['net_qty'] 
                    }

            # Calculate metrics
            self.total_premium_received = (short_ce['avg_price'] * abs(short_ce['net_qty']) + 
                                        short_pe['avg_price'] * abs(short_pe['net_qty']))
            self.premium_target = self.total_premium_received / 2.0

            # Recalculate ranges
            self.lower_range = self.entry_spot_price - short_pe['avg_price']
            self.higher_range = self.entry_spot_price + short_ce['avg_price']
            self.max_premium = max(short_ce['avg_price'], short_pe['avg_price'])

            # Set dates
            self.entry_date = ISTTimeUtils.now().date()
            self.expiry_date = self._get_expiry_date()

            logger.info(f"Strategy RECOVERED with protective hedges")
            logger.info(f"Original Premium: {self.total_premium_received:.2f}")
            logger.info(f"Profit Target: {self.premium_target:.2f}")
            logger.info(f"Entry Spot: {self.entry_spot_price:.2f}")
            logger.info(f"Range: {self.lower_range:.0f} - {self.higher_range:.0f}")
            
            # Get current market data for monitoring
            self._get_spot_price()
            
            # Register with PositionManager
            self.register_strategy_positions()
                
            self._log_state("ACTIVE", "Full strategy recovered with hedges")
            return True

        except Exception as e:
            logger.error(f"Error during strategy recovery: {e}", exc_info=True)
            return False

    def _get_strategy_positions_from_broker(self):
        try:
            broker_positions = self._get_broker_positions()
            strategy_positions = []
            
            for pos in broker_positions:
                symbol = pos.get('tsym', '')
                if 'NIFTY' in symbol and ('CE' in symbol or 'PE' in symbol):
                    strategy_positions.append({
                        'symbol': symbol,
                        'token': pos.get('token', ''),
                        'net_qty': int(float(pos.get('netqty', 0))),
                        'avg_price': float(pos.get('netupldprc', 0)),
                        'ltp': float(pos.get('lp', 0))
                    })
            
            return strategy_positions
        except Exception as e:
            logger.error(f"Error getting strategy positions from broker: {e}")
            return []

    def register_strategy_positions(self, position_manager=None):
        try:
            target_manager = position_manager or self.position_manager
            if not target_manager:
                logger.error("No position manager available for registration")
                return False

            spot_price = self._get_current_spot_price()
            
            for leg, pos in self.positions.items():
                if pos and pos.get('symbol') and pos.get('net_qty', 0) != 0:
                    key = f"{pos['symbol']}_{pos.get('token', '')}"
                    strategy_name = "Monthly Straddle"
                    if "buy" in leg:
                        strategy_name += " Hedge"
                    
                    target_manager._strategy_symbol_token_map[key] = {
                        'strategy_name': strategy_name,
                        'spot_price': spot_price,
                        'timestamp': datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
                    }
            
            target_manager._save_strategy_mapping()
            logger.info("Registered Monthly Straddle strategy positions with PositionManager")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register strategy positions: {str(e)}")
            return False

    def execute_strategy(self):
        logger.info("Executing Monthly Straddle strategy")
        self._run_strategy_cycle()

    def on_execute_strategy_clicked(self):
        try:
            ct = ISTTimeUtils.current_time()
            if not self._is_entry_day():
                logger.warning("Manual execute clicked - not an entry day.")
                return
            if ct < time(9, 15) or ct > time(15, 15):
                logger.warning("Manual execute clicked - outside trading window.")
                return
            logger.info("Manual strategy execution requested.")
            self._run_strategy_cycle()
        except Exception:
            logger.exception("Error during manual execute click")

    # ===== Core Strategy Logic =====
    def _get_primary_client(self):
        try:
            if self.client_manager and hasattr(self.client_manager, "clients") and self.client_manager.clients:
                return self.client_manager.clients[0][2]
        except Exception:
            logger.exception("Error getting primary client")
        return None

    def _validate_clients(self):
        client = self._get_primary_client()
        valid = client is not None
        if not valid:
            logger.warning("No active client available")
        return valid

    @safe_log("Order execution")
    def _execute_order(self, client, symbol, token, action, quantity, remark=""):
        try:
            client.place_order(
                buy_or_sell=action[0].upper(),
                product_type="M",
                exchange="NFO",
                tradingsymbol=symbol,
                quantity=quantity,
                discloseqty=0,
                price_type="MKT",
                price=0,
                trigger_price=0,
                retention="DAY",
                remarks=remark
            )
            logger.info(f"Order executed: {action} {symbol} qty={quantity} remark={remark}")
            return True
        except Exception as e:
            logger.error(f"Order failed: {action} {symbol} - {e}")
            return False

    def _is_entry_day(self):
        try:
            today = ISTTimeUtils.now().date()
            if not os.path.exists(self.nfo_symbols_path):
                logger.error("NFO_symbols.txt not found")
                return False

            df = pd.read_csv(self.nfo_symbols_path)
            df = df[df["Instrument"].str.strip() == "OPTIDX"]
            df["Expiry"] = pd.to_datetime(df["Expiry"], format="%d-%b-%Y", errors="coerce")
            
            if df["Expiry"].isnull().all():
                logger.error("No valid expiry date parsed")
                return False

            current_month_start = today.replace(day=1)
            current_month_period = pd.Timestamp(current_month_start).to_period("M")
            current_month_expiries = df[df["Expiry"].dt.to_period("M") == current_month_period]

            if current_month_expiries.empty:
                logger.warning("No current month expiries found")
                return False

            last_tuesday = None
            for exp in sorted(current_month_expiries["Expiry"].dt.date.unique()):
                if exp.weekday() == 1:  # Tuesday
                    last_tuesday = exp if (last_tuesday is None or exp > last_tuesday) else last_tuesday

            if not last_tuesday:
                logger.warning("No Tuesday expiry found for current month")
                return False

            next_trading_day = last_tuesday + timedelta(days=1)
            while next_trading_day.weekday() >= 5:
                next_trading_day += timedelta(days=1)

            logger.debug("Last Tuesday expiry: %s; Next trading day: %s; Today: %s",
                        last_tuesday, next_trading_day, today)
            return today == next_trading_day

        except Exception as e:
            logger.exception("Error computing entry day: %s", e)
            return False

    def _get_expiry_date(self):
        try:
            today = ISTTimeUtils.now().date()
            if not os.path.exists(self.nfo_symbols_path):
                logger.error("NFO_symbols.txt not found")
                return None

            df = pd.read_csv(self.nfo_symbols_path)
            df = df[df["Instrument"].str.strip() == "OPTIDX"]
            df["Expiry"] = pd.to_datetime(df["Expiry"], format="%d-%b-%Y", errors="coerce")

            if df["Expiry"].isnull().all():
                logger.error("No valid expiry parsing")
                return None

            if today.day < 15:
                target_month = today.replace(day=1)
            else:
                target_month = (today.replace(day=1) + timedelta(days=32)).replace(day=1)

            target_period = pd.Timestamp(target_month).to_period("M")
            month_expiries = df[df["Expiry"].dt.to_period("M") == target_period]

            if month_expiries.empty:
                logger.error(f"No expiries found for month {target_month}")
                return None

            expiry_date = month_expiries["Expiry"].max().date()
            logger.info("Selected expiry date: %s", expiry_date)
            return expiry_date

        except Exception as e:
            logger.exception("Error getting expiry date: %s", e)
            return None

    @safe_log("LTP fetch")
    def _get_ltp(self, client, token_or_symbol):
        try:
            quote = client.get_quotes("NFO", str(token_or_symbol))
            if quote and quote.get("stat") == "Ok" and "lp" in quote:
                ltp = float(quote["lp"])
                logger.debug("LTP for %s: %s", token_or_symbol, ltp)
                return ltp
            logger.warning("Invalid quote response for token %s", token_or_symbol)
            return 0.0
        except Exception as e:
            logger.error("Error fetching LTP for %s: %s", token_or_symbol, e)
            return 0.0

    @safe_log("Spot price fetch")
    def _get_spot_price(self):
        try:
            if not self._validate_clients():
                return False
            client = self._get_primary_client()
            quote = client.get_quotes("NSE", "26000")
            if not quote or quote.get("stat") != "Ok":
                logger.error("Failed to get NIFTY quote")
                return False

            self.spot_price = float(quote.get("lp", 0.0))
            logger.info("Current NIFTY spot: %.2f", self.spot_price)
            return True
        except Exception as e:
            logger.exception("Error in _get_spot_price: %s", e)
            return False

    def _get_current_spot_price(self):
        try:
            if hasattr(self.client_manager, 'clients') and self.client_manager.clients:
                client = self.client_manager.clients[0][2]
                quote = client.get_quotes('NSE', '26000')
                
                if not quote or quote.get('stat') != 'Ok':
                    logger.error(f"Failed to get NIFTY quote: {quote}")
                    return 0.0
                    
                spot_price = float(quote.get('lp', 0))
                logger.info(f"Current NIFTY spot price: {spot_price}")
                return spot_price
            return 0.0
        except Exception as e:
            logger.error(f"Failed to get spot price: {str(e)}")
            return 0.0

    def _select_strikes(self, expiry_date):
        try:
            if not os.path.exists(self.nfo_symbols_path):
                logger.error("NFO_symbols.txt not found")
                return None

            df = pd.read_csv(self.nfo_symbols_path)
            df = df[df["Instrument"].str.strip() == "OPTIDX"]
            df["Expiry"] = df["Expiry"].str.strip().str.upper()

            expiry_str = expiry_date.strftime("%d-%b-%Y").upper()
            expiry_options = df[(df["Expiry"] == expiry_str) & (df["Symbol"].str.strip() == "NIFTY")]

            if expiry_options.empty:
                logger.error("No options rows found for expiry %s", expiry_str)
                return None

            if not self._validate_clients():
                return None
            client = self._get_primary_client()

            atm_strike = None
            min_price_diff = float("inf")
            ce_ltp = pe_ltp = 0.0
            ce_symbol = pe_symbol = None
            ce_token = pe_token = None

            min_strike = int(self.spot_price - 500)
            max_strike = int(self.spot_price + 500)

            uniq_strikes = sorted(expiry_options["StrikePrice"].unique())
            for strike in uniq_strikes:
                if strike < min_strike or strike > max_strike:
                    continue

                ce_data = expiry_options[(expiry_options["StrikePrice"] == strike) & (expiry_options["OptionType"] == "CE")]
                pe_data = expiry_options[(expiry_options["StrikePrice"] == strike) & (expiry_options["OptionType"] == "PE")]

                if ce_data.empty or pe_data.empty:
                    continue

                ce_token_tmp = ce_data.iloc[0]["Token"]
                pe_token_tmp = pe_data.iloc[0]["Token"]

                ce_ltp_tmp = self._get_ltp(client, ce_token_tmp)
                pe_ltp_tmp = self._get_ltp(client, pe_token_tmp)

                if not (10 <= ce_ltp_tmp <= 25 and 10 <= pe_ltp_tmp <= 25):
                    continue

                price_diff = abs(ce_ltp_tmp - pe_ltp_tmp)
                if price_diff < min_price_diff:
                    min_price_diff = price_diff
                    atm_strike = strike
                    ce_ltp = ce_ltp_tmp
                    pe_ltp = pe_ltp_tmp
                    ce_symbol = ce_data.iloc[0]["TradingSymbol"]
                    pe_symbol = pe_data.iloc[0]["TradingSymbol"]
                    ce_token = ce_token_tmp
                    pe_token = pe_token_tmp

            if atm_strike is None:
                logger.error("No ATM strike found that meets LTP criteria (10..25)")
                return None

            total_premium = (ce_ltp + pe_ltp) * LOT_SIZE
            premium_target = total_premium / 2.0

            ce_protective_strike, ce_protective_symbol, ce_protective_token = self._find_protective_strike(
                expiry_options, uniq_strikes, atm_strike, "CE", client
            )
            pe_protective_strike, pe_protective_symbol, pe_protective_token = self._find_protective_strike(
                expiry_options, uniq_strikes, atm_strike, "PE", client
            )

            if not ce_protective_strike or not pe_protective_strike:
                logger.error("Could not find suitable protective strikes")
                return None

            logger.info("Selected strikes: ATM=%s CE_ltp=%.2f PE_ltp=%.2f ; CE_protect=%s PE_protect=%s",
                        atm_strike, ce_ltp, pe_ltp, ce_protective_strike, pe_protective_strike)

            self.total_premium_received = total_premium
            self.premium_target = premium_target

            return {
                "atm_strike": atm_strike,
                "ce_sell": {"symbol": ce_symbol, "token": ce_token, "ltp": ce_ltp},
                "pe_sell": {"symbol": pe_symbol, "token": pe_token, "ltp": pe_ltp},
                "ce_protective_strike": ce_protective_strike,
                "ce_buy": {"symbol": ce_protective_symbol, "token": ce_protective_token},
                "pe_protective_strike": pe_protective_strike,
                "pe_buy": {"symbol": pe_protective_symbol, "token": pe_protective_token}
            }

        except Exception as e:
            logger.exception("Error in _select_strikes: %s", e)
            return None

    def _find_protective_strike(self, expiry_options, uniq_strikes, atm_strike, option_type, client):
        strike = None
        symbol = None
        token = None

        if option_type == "CE":
            search_strikes = sorted([s for s in uniq_strikes if s > atm_strike], reverse=True)
        else:
            search_strikes = sorted([s for s in uniq_strikes if s < atm_strike])

        for s in search_strikes:
            rows = expiry_options[(expiry_options["StrikePrice"] == s) & (expiry_options["OptionType"] == option_type)]
            if rows.empty:
                continue

            token_tmp = rows.iloc[0]["Token"]
            ltp_tmp = self._get_ltp(client, token_tmp)
            if 10 <= ltp_tmp <= 25:
                strike = s
                symbol = rows.iloc[0]["TradingSymbol"]
                token = token_tmp
                break

        return strike, symbol, token

    def _place_straddle_orders(self, expiry_date, strikes, include_hedges=True):
        try:
            if not self._validate_clients():
                logger.error("_place_straddle_orders aborted - no client")
                return False
                
            client = self._get_primary_client()
            qty = LOT_SIZE

            logger.info("Placing straddle orders. include_hedges=%s", include_hedges)

            ce_sell_ok = self._execute_order(client, strikes["ce_sell"]["symbol"], strikes["ce_sell"]["token"], "SELL", qty, "ATM CE Sell")
            pe_sell_ok = self._execute_order(client, strikes["pe_sell"]["symbol"], strikes["pe_sell"]["token"], "SELL", qty, "ATM PE Sell")

            ce_buy_ok = pe_buy_ok = True
            if include_hedges:
                ce_buy_ok = self._execute_order(client, strikes["ce_buy"]["symbol"], strikes["ce_buy"]["token"], "BUY", qty, "Protective CE Buy")
                pe_buy_ok = self._execute_order(client, strikes["pe_buy"]["symbol"], strikes["pe_buy"]["token"], "BUY", qty, "Protective PE Buy")

            if ce_sell_ok and pe_sell_ok and ce_buy_ok and pe_buy_ok:
                position_data = self._create_position_data(strikes, client, qty, include_hedges)
                
                for leg, data in position_data.items():
                    self.positions[leg] = data

                ce_premium = self.positions["ce_sell"]["entry_price"]
                pe_premium = self.positions["pe_sell"]["entry_price"]
                
                self.total_premium_received = (ce_premium + pe_premium) * qty
                self.premium_target = self.total_premium_received / 2.0
                self.lower_range = self.spot_price - pe_premium
                self.higher_range = self.spot_price + ce_premium
                self.max_premium = max(ce_premium, pe_premium)

                self.expiry_date = expiry_date
                self.entry_date = ISTTimeUtils.now().date()
                self.entry_spot_price = self.spot_price

                logger.info("Orders placed successfully. Ranges: %.2f - %.2f ; max_premium=%.2f",
                            self.lower_range, self.higher_range, self.max_premium)

                # Register with PositionManager
                self.register_strategy_positions()

                self._log_state(
                    status="ACTIVE", 
                    comments="Positions placed with hedges" if include_hedges else "Positions placed without hedges",
                    expiry_date=expiry_date,
                    entry_date=self.entry_date
                )

                return True

            logger.warning("Partial order failure. Rolling back successful legs if any.")
            self._rollback_orders(client, qty, ce_sell_ok, pe_sell_ok, ce_buy_ok, pe_buy_ok, strikes, include_hedges)
            return False

        except Exception as e:
            logger.exception("Error placing straddle orders: %s", e)
            return False

    def _create_position_data(self, strikes, client, qty, include_hedges):
        position_data = {
            'ce_sell': {
                'symbol': strikes["ce_sell"]["symbol"],
                'token': strikes["ce_sell"]["token"],
                'entry_price': strikes["ce_sell"]["ltp"],
                'net_qty': -qty,
                'strike': strikes["atm_strike"],
                'option_type': 'CE'
            },
            'pe_sell': {
                'symbol': strikes["pe_sell"]["symbol"],
                'token': strikes["pe_sell"]["token"],
                'entry_price': strikes["pe_sell"]["ltp"],
                'net_qty': -qty,
                'strike': strikes["atm_strike"],
                'option_type': 'PE'
            }
        }
        
        if include_hedges:
            position_data.update({
                'ce_buy': {
                    'symbol': strikes["ce_buy"]["symbol"],
                    'token': strikes["ce_buy"]["token"],
                    'entry_price': self._get_ltp(client, strikes["ce_buy"]["token"]),
                    'net_qty': qty,
                    'strike': strikes["ce_protective_strike"],
                    'option_type': 'CE'
                },
                'pe_buy': {
                    'symbol': strikes["pe_buy"]["symbol"],
                    'token': strikes["pe_buy"]["token"],
                    'entry_price': self._get_ltp(client, strikes["pe_buy"]["token"]),
                    'net_qty': qty,
                    'strike': strikes["pe_protective_strike"],
                    'option_type': 'PE'
                }
            })
        
        return position_data

    def _rollback_orders(self, client, qty, ce_sell_ok, pe_sell_ok, ce_buy_ok, pe_buy_ok, strikes, include_hedges):
        rollback_actions = []
        
        if ce_sell_ok:
            rollback_actions.append(('BUY', strikes["ce_sell"]["symbol"], strikes["ce_sell"]["token"], "Cancel CE Sell"))
        if pe_sell_ok:
            rollback_actions.append(('BUY', strikes["pe_sell"]["symbol"], strikes["pe_sell"]["token"], "Cancel PE Sell"))
        if include_hedges and ce_buy_ok:
            rollback_actions.append(('SELL', strikes["ce_buy"]["symbol"], strikes["ce_buy"]["token"], "Cancel CE Buy"))
        if include_hedges and pe_buy_ok:
            rollback_actions.append(('SELL', strikes["pe_buy"]["symbol"], strikes["pe_buy"]["token"], "Cancel PE Buy"))
            
        for action, symbol, token, remark in rollback_actions:
            self._execute_order(client, symbol, token, action, qty, remark)

    def _check_and_execute_strategy(self):
        try:
            ct = ISTTimeUtils.current_time()
            if self._is_entry_day() and ct.hour == 9 and ct.minute == 20 and self.state == "WAITING":
                logger.info("Scheduled entry condition matched (9:20). Running strategy cycle.")
                self._run_strategy_cycle()

            if time(9, 30) <= ct <= time(15, 15) and self.state == "ACTIVE":
                self._monitor_positions()
        except Exception:
            logger.exception("Error in scheduled check and execute")

    def _run_strategy_cycle(self):
        try:
            if not self._get_spot_price():
                logger.error("Aborting strategy cycle - cannot fetch spot")
                return False

            expiry = self._get_expiry_date()
            if not expiry:
                logger.error("Aborting strategy cycle - cannot obtain expiry")
                return False
            self.expiry_date = expiry
            self.entry_date = ISTTimeUtils.now().date()
            self.entry_spot_price = self.spot_price

            strikes = self._select_strikes(self.expiry_date)
            if not strikes:
                logger.error("Aborting strategy cycle - strike selection failed")
                return False

            placed = self._place_straddle_orders(self.expiry_date, strikes, include_hedges=True)
            if placed:
                self.state = "ACTIVE"
                logger.info("Strategy state set to ACTIVE")
                self._log_state("ACTIVE", "Positions placed")
                return True
            else:
                logger.error("Order placement failed during strategy cycle")
                return False

        except Exception:
            logger.exception("Error running strategy cycle")
            return False

    def _calculate_current_pnl(self):
        try:
            client = self._get_primary_client()
            if not client:
                return 0.0

            total_pnl = 0.0
            for pos_type, pos in self.positions.items():
                if pos and "sell" in pos_type:
                    quote = client.get_quotes("NFO", str(pos["token"]))
                    if quote and quote.get("stat") == "Ok" and "lp" in quote:
                        current_price = float(quote["lp"])
                        pnl = (pos["entry_price"] - current_price) * LOT_SIZE
                        total_pnl += pnl
            logger.debug("Calculated current PnL: %.2f", total_pnl)
            return total_pnl
        except Exception:
            logger.exception("Error calculating current PnL")
            return 0.0

    def _check_exit_conditions(self):
        try:
            current_pnl = self._calculate_current_pnl()
            logger.debug("Checking exit conditions: pnl=%.2f target=%.2f", current_pnl, self.premium_target)
            
            if current_pnl >= self.premium_target:
                logger.info("Exit condition satisfied: premium target reached")
                return True

            if self.expiry_date:
                days_to_expiry = (self.expiry_date - ISTTimeUtils.now().date()).days
                if days_to_expiry <= 7:
                    logger.info("Exit condition satisfied: within last week of expiry (days=%s)", days_to_expiry)
                    return True

            if self.entry_spot_price is not None and self.max_premium is not None:
                if (self.spot_price > self.entry_spot_price + self.max_premium) or \
                (self.spot_price < self.entry_spot_price - self.max_premium):
                    logger.info("Exit condition satisfied: market moved beyond premium range")
                    self._show_alert_message(
                        "Market Movement Alert", 
                        f"Spot price moved beyond premium range!\nEntry Spot: ₹{self.entry_spot_price:.2f}\nCurrent Spot: ₹{self.spot_price:.2f}\nMax Premium: ₹{self.max_premium:.2f}\nConsidering adjustment.",
                        "warning"
                    )
                    return True

            return False
        except Exception:
            logger.exception("Error checking exit conditions")
            return False

    def _monitor_positions(self):
        try:
            if not self._get_spot_price():
                return
                
            client = self._get_primary_client()
            if client:
                for leg in self.positions:
                    if self.positions[leg]:
                        current_ltp = self._get_ltp(client, self.positions[leg]['token'])
                        if current_ltp > 0:
                            self.positions[leg]['ltp'] = current_ltp
            
            if self._check_exit_conditions():
                self._exit_all_positions()
                return
                
            self._log_state(
                status=self.state,
                comments="Regular monitoring update",
                expiry_date=self.expiry_date,
                entry_date=self.entry_date
            )
            
        except Exception as e:
            logger.error(f"Monitoring error: {e}")

    def _exit_all_positions(self):
        try:
            if not self._validate_clients():
                logger.error("Exit aborted - no client")
                return False
                
            client = self._get_primary_client()
            qty = LOT_SIZE
            
            exit_success = True
            
            for pos_type, pos in self.positions.items():
                if pos:
                    if "sell" in pos_type:
                        action = "BUY"
                        action_desc = f"Exit {pos_type}"
                    else:
                        action = "SELL" 
                        action_desc = f"Exit {pos_type}"
                    
                    success = self._execute_order(client, pos["symbol"], pos["token"], action, qty, action_desc)
                    if success:
                        logger.info("Exited position %s", pos_type)
                        self.positions[pos_type] = None
                    else:
                        exit_success = False
                        logger.error("Failed to exit position %s", pos_type)
            
            if exit_success:
                self.state = "COMPLETED"
                self._log_state("COMPLETED", "ALL positions exited (sell legs + protective buys)")
                logger.info("ALL positions exited successfully")
                return True
                
            logger.warning("Some positions failed to exit")
            return False
            
        except Exception as e:
            logger.exception("Error exiting positions: %s", e)
            return False

    # ===== Utility Methods =====
    def _parse_dates_from_state(self, latest_state):
        def parse_date_dynamic(date_str, default_date):
            if pd.isna(date_str) or not date_str:
                return default_date

            date_str = str(date_str).strip().upper()

            if len(date_str) == 9 and date_str[6:].isdigit():
                year_full = 2000 + int(date_str[6:])
                date_str = date_str[:6] + str(year_full)

            for fmt in ("%d-%b-%Y", "%Y-%m-%d"):
                try:
                    return datetime.strptime(date_str, fmt).date()
                except ValueError:
                    continue

            logger.warning(f"Could not parse date: {date_str}, using default {default_date}")
            return default_date

        try:
            today = pd.Timestamp.now().date()
            last_day = monthrange(today.year, today.month)[1]
            default_expiry = date(today.year, today.month, last_day)

            self.expiry_date = parse_date_dynamic(latest_state.get('expiry_date'), default_expiry)
            self.entry_date = parse_date_dynamic(latest_state.get('entry_date'), today)

        except Exception as e:
            logger.warning(f"Error parsing dates: {e}")
            self.expiry_date = date(today.year, today.month, last_day)
            self.entry_date = pd.Timestamp.now().date()

    def _find_latest_state_file(self):
        try:
            if os.path.exists(self.current_state_file):
                return self.current_state_file
                
            state_files = []
            for file in os.listdir(self.log_dir):
                if file.endswith('_monthly_straddle_state.csv'):
                    state_files.append(file)
            
            if not state_files:
                logger.info("No monthly straddle state files found")
                return None
                
            state_files.sort(reverse=True)
            latest_file = state_files[0]
            latest_file_path = os.path.join(self.log_dir, latest_file)
            
            logger.info(f"Using latest state file: {latest_file}")
            return latest_file_path
            
        except Exception as e:
            logger.error(f"Error finding latest state file: {e}")
            return None

    def _get_spot_price_from_csv(self):
        try:
            csv_file = f"{ISTTimeUtils.current_date_str()}_strategy_mapping.csv"
            csv_path = os.path.join(self.log_dir, csv_file)
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                straddle_entries = df[df['strategy'] == 'Monthly Straddle']
                if not straddle_entries.empty:
                    spot_price = straddle_entries['spot_price'].iloc[0]
                    logger.info(f"Retrieved spot price from CSV: {spot_price}")
                    return spot_price
        except Exception as e:
            logger.warning(f"Could not read spot price from CSV: {e}")
        return None

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

    def _log_state(self, status, comments="", **kwargs):
        try:
            current_prices = {}
            client = self._get_primary_client()
            if client:
                for leg in ['ce_sell', 'pe_sell', 'ce_buy', 'pe_buy']:
                    if self.positions[leg]:
                        token = self.positions[leg].get('token')
                        if token:
                            try:
                                current_prices[leg] = self._get_ltp(client, token)
                            except Exception as e:
                                logger.warning(f"Failed to get LTP for {leg}: {e}")
                                current_prices[leg] = self.positions[leg].get('entry_price', 0.0)
                        else:
                            current_prices[leg] = self.positions[leg].get('entry_price', 0.0)
            
            state_data = {
                "timestamp": ISTTimeUtils.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": status,
                "comments": comments,
                "spot_price": self.spot_price,
                "entry_spot_price": self.entry_spot_price,
                "total_premium": self.total_premium_received,
                "premium_target": self.premium_target,
                "lower_range": self.lower_range,
                "higher_range": self.higher_range,
                "max_premium": self.max_premium,
                "expiry_date": self.expiry_date.strftime("%d-%b-%Y") if self.expiry_date else None,
                "entry_date": self.entry_date.strftime("%d-%b-%Y") if self.entry_date else None,
            }
            
            position_legs = ['ce_sell', 'pe_sell', 'ce_buy', 'pe_buy']
            for leg in position_legs:
                pos = self.positions[leg]
                if pos:
                    state_data.update({
                        f"{leg}_symbol": pos.get("symbol"),
                        f"{leg}_token": pos.get("token", ""),
                        f"{leg}_entry": pos.get("entry_price", 0.0),
                        f"{leg}_current": current_prices.get(leg, 0.0),
                        f"{leg}_qty": pos.get("net_qty", 0),
                        f"{leg}_strike": pos.get("strike"),
                        f"{leg}_option_type": pos.get("option_type")
                    })
                else:
                    state_data.update({
                        f"{leg}_symbol": None,
                        f"{leg}_token": "",
                        f"{leg}_entry": 0.0,
                        f"{leg}_current": 0.0,
                        f"{leg}_qty": 0,
                        f"{leg}_strike": None,
                        f"{leg}_option_type": None
                    })
            
            state_data.update(kwargs)
            
            file_exists = os.path.exists(self.current_state_file)
            pd.DataFrame([state_data]).to_csv(self.current_state_file, mode='a', 
                                            header=not file_exists, index=False)
            
            logger.debug(f"State logged: {status} - {comments}")
            
        except Exception as e:
            logger.error(f"Failed to log state: {str(e)}")

    def _show_alert_message(self, title, message, level="info"):
        try:
            msg_box = QMessageBox()
            msg_box.setWindowTitle(title)
            msg_box.setText(message)
            
            if level == "warning":
                msg_box.setIcon(QMessageBox.Warning)
            elif level == "critical":
                msg_box.setIcon(QMessageBox.Critical)
            else:
                msg_box.setIcon(QMessageBox.Information)
                
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec_()
            
            logger.info(f"User alert: {title} - {message}")
            
        except Exception as e:
            logger.error(f"Failed to show alert: {e}")

    def cleanup(self):
        logger.info("Performing cleanup for MonthlyStraddleStrategy")
        try:
            if hasattr(self.strategy_timer, "isActive") and self.strategy_timer.isActive():
                self.strategy_timer.stop()
                logger.debug("Strategy timer stopped")
        except Exception:
            logger.exception("Error stopping strategy timer")

        self.state = "WAITING"
        self.positions = {k: None for k in self.positions}
        self.spot_price = None
        self.entry_spot_price = None
        self.lower_range = None
        self.higher_range = None
        self.total_premium_received = 0.0
        self.premium_target = 0.0
        self.max_premium = 0.0

        logger.info("Cleanup complete")