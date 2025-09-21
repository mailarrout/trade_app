import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import time as sleep_time
import threading
import csv
import os
from modules.option_loader import OptionLoader
import winsound
import pytz
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QTableWidgetItem, QPushButton, QMessageBox
from PyQt5.QtGui import QColor, QFont
import requests
import math
from scipy.stats import norm
from zoneinfo import ZoneInfo
import logging

# Constants
LOT_SIZE = 75
IST_TIMEZONE = pytz.timezone('Asia/Kolkata')
DELTA_THRESHOLD = 20
MAX_STRIKE_SEARCH_RANGE = 200  # ±200 points from spot for optimal strike
HEDGE_PRICE_RANGE = (10, 25)   # Price range for hedge positions
HEDGE_MAX_SEARCH_DISTANCE = 1000  # Max points to search for hedge strikes
DELTA_UPDATE_MINUTES = 5  # Update delta every 5 minutes
ENTRY_TIME = time(1, 15)  # 9:15 AM IST
EXIT_TIME = time(15, 30)  # 3:30 PM IST
STRATEGY_CHECK_INTERVAL_MS = 60000  # Check every minute
GREEK_UPDATE_INTERVAL_MS = 300000  # Update every 5 seconds
RISK_FREE_RATE = 0.07  # 7% risk-free rate
DEFAULT_TIME_TO_EXPIRY = 0.08  # ~1 month
NFO_SYMBOLS_FILE = "NFO_symbols.txt"
INDEX_SYMBOL = "NIFTY"
INSTRUMENT_TYPE = "OPTIDX"
FUTURES_TYPE = "FUTIDX"
ORDER_PRODUCT_TYPE = "M"
ORDER_EXCHANGE = "NFO"
ORDER_PRICE_TYPE = "MKT"
ORDER_RETENTION = "DAY"
ORDER_REMARKS = "StraddleStrategy"

# Get logger for this module
logger = logging.getLogger(__name__)

class DailyStraddleStrategy:
    
    def __init__(self, ui, client_manager, option_loader):
        logger.info("Initializing DailyStraddleStrategy")
        self.ui = ui
        self.client_manager = client_manager
        self.option_loader = option_loader
        self.positions = {}
        self.is_running = False
        self.strategy_executed = False
        self.monitor_thread = None
        
        # Set up log directory
        self.log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Strategy data file in log directory
        today_str = datetime.now(IST_TIMEZONE).strftime('%Y%m%d')
        self.strategy_data_file = os.path.join(
            self.log_dir, 
            f"{today_str}-strategy-intraday-straddle.csv"
        )
        
        # Reset strategy_executed if it's a new day
        self.strategy_executed_today = today_str
        
        # Set up periodic strategy check timer
        self.strategy_timer = QTimer()
        self.strategy_timer.timeout.connect(self.check_and_start_strategy)
        self.strategy_timer.start(STRATEGY_CHECK_INTERVAL_MS)
        
        logger.info("Daily straddle strategy initialized with periodic checking")

        self.greek_timer = QTimer()
        self.greek_timer.timeout.connect(self._update_positions_greek)
        self.greek_timer.start(GREEK_UPDATE_INTERVAL_MS)
        
        logger.debug("Timers setup completed")
    
    def check_and_start_strategy(self):
        """Check every minute if strategy should be started"""
        try:
            today_str = datetime.now(IST_TIMEZONE).strftime('%Y%m%d')
            if today_str != self.strategy_executed_today:
                self.strategy_executed = False
                self.strategy_executed_today = today_str
                logger.info(f"New trading day detected: {today_str}, resetting strategy flag")
            
            if self.is_running or self.strategy_executed:
                logger.debug("Strategy already running or executed today - skipping")
                return
                
            current_time = datetime.now(IST_TIMEZONE).time()
            if ENTRY_TIME <= current_time <= EXIT_TIME:
                logger.info(f"Strategy time window active ({current_time.strftime('%H:%M')} IST)")
                if self.execute_strategy():
                    self.strategy_executed = True
            else:
                logger.debug(f"Outside strategy time window: {current_time.strftime('%H:%M')} IST")
                
        except Exception as e:
            logger.error(f"Error in strategy check: {str(e)}")
    
    def get_spot_price(self):
        """Get current NIFTY spot price from futures data"""
        try:
            logger.debug("Fetching spot price")
            
            if not os.path.exists(NFO_SYMBOLS_FILE):
                logger.error(f"{NFO_SYMBOLS_FILE} file not found")
                return 0
            
            df = pd.read_csv(NFO_SYMBOLS_FILE)
            df = df[(df["Instrument"].str.strip() == FUTURES_TYPE) & 
                    (df["Symbol"].str.strip() == INDEX_SYMBOL)]
            
            if df.empty:
                logger.error(f"No futures data found for {INDEX_SYMBOL}")
                return 0
            
            token = str(df.iloc[0]['Token'])
            client = self.client_manager.clients[0][2]
            quote = client.get_quotes(ORDER_EXCHANGE, token)

            if not quote or quote.get('stat') != 'Ok':
                logger.error(f"Failed to get {INDEX_SYMBOL} futures quote")
                return 0

            spot_price = float(quote.get('lp', 0))
            
            if spot_price > 0:
                logger.info(f"Spot price retrieved: {spot_price:.2f}")
                return spot_price
            else:
                logger.error("Invalid spot price received")
                return 0
                
        except Exception as e:
            logger.error(f"Error getting spot price: {str(e)}")
            return 0
    
    def get_expiry_date(self):
        """Get the current expiry date from UI"""
        expiry_date = self.ui.ExpiryListDropDown.currentText()
        logger.debug(f"Retrieved expiry date from UI: {expiry_date}")
        return expiry_date
    
    def get_option_ltp(self, symbol, token):
        """Get LTP for a specific option symbol"""
        try:
            logger.debug(f"Getting LTP for {symbol}")
            client = self.client_manager.clients[0][2]
            quote = client.get_quotes(ORDER_EXCHANGE, str(token))
            if quote and quote.get('stat') == 'Ok':
                ltp = float(quote.get('lp', 0))
                logger.debug(f"LTP for {symbol}: {ltp:.2f}")
                return ltp
            logger.warning(f"No valid quote for {symbol}")
            return 0
        except Exception as e:
            logger.error(f"Error getting LTP for {symbol}: {str(e)}")
            return 0
    
    def find_optimal_straddle_strike(self, spot_price, expiry_date):
        """Find strike with minimum CE-PE price difference within ±200 range"""
        try:
            logger.info(f"Finding optimal straddle strike for spot: {spot_price}, expiry: {expiry_date}")
            
            if not os.path.exists(NFO_SYMBOLS_FILE):
                logger.error(f"{NFO_SYMBOLS_FILE} file not found")
                return None, None, None
            
            df = pd.read_csv(NFO_SYMBOLS_FILE)
            
            expiry_dt = datetime.strptime(expiry_date, '%d-%b-%Y')
            expiry_str_file = expiry_dt.strftime("%d-%b-%Y").upper()
            
            df = df[
                (df["Instrument"].str.strip() == INSTRUMENT_TYPE) & 
                (df["Symbol"].str.strip() == INDEX_SYMBOL) & 
                (df["Expiry"].str.strip().str.upper() == expiry_str_file)
            ]
            
            if df.empty:
                logger.error(f"No {INDEX_SYMBOL} options found for expiry {expiry_str_file}")
                return None, None, None
            
            rounded_spot = round(spot_price / 50) * 50
            logger.info(f"Rounded spot price: {rounded_spot}")
            
            optimal_strike = None
            min_difference = float('inf')
            ce_ltp = None
            pe_ltp = None
            
            min_strike = rounded_spot - MAX_STRIKE_SEARCH_RANGE
            max_strike = rounded_spot + MAX_STRIKE_SEARCH_RANGE
            
            logger.info(f"Searching strikes between {min_strike} and {max_strike} (±{MAX_STRIKE_SEARCH_RANGE} points)")
            
            strikes_in_range = df[
                (df['StrikePrice'] >= min_strike) & 
                (df['StrikePrice'] <= max_strike)
            ]['StrikePrice'].unique()
            
            strikes_in_range = sorted(strikes_in_range)
            logger.info(f"Found {len(strikes_in_range)} strikes in range")
            
            for strike in strikes_in_range:
                ce_data = df[(df['StrikePrice'] == strike) & (df['OptionType'] == 'CE')]
                pe_data = df[(df['StrikePrice'] == strike) & (df['OptionType'] == 'PE')]
                
                if not ce_data.empty and not pe_data.empty:
                    ce_symbol = ce_data.iloc[0]['TradingSymbol']
                    pe_symbol = pe_data.iloc[0]['TradingSymbol']
                    ce_token = ce_data.iloc[0]['Token']
                    pe_token = pe_data.iloc[0]['Token']
                    
                    ce_current_ltp = self.get_option_ltp(ce_symbol, ce_token)
                    pe_current_ltp = self.get_option_ltp(pe_symbol, pe_token)
                    
                    if ce_current_ltp > 0 and pe_current_ltp > 0:
                        difference = abs(ce_current_ltp - pe_current_ltp)
                        
                        if difference < min_difference:
                            min_difference = difference
                            optimal_strike = strike
                            ce_ltp = ce_current_ltp
                            pe_ltp = pe_current_ltp
            
            if optimal_strike:
                logger.info(f"Selected optimal strike: {optimal_strike}")
                logger.info(f"CE LTP: {ce_ltp:.2f}, PE LTP: {pe_ltp:.2f}, Difference: {min_difference:.2f}")
            else:
                logger.warning("No optimal strike found in range")
                
            return optimal_strike, ce_ltp, pe_ltp
            
        except Exception as e:
            logger.error(f"Error finding optimal strike: {str(e)}")
            return None, None, None
    
    def find_hedge_strike_directional(self, main_strike, option_type, expiry_date):
        """Find hedge strike by searching directionally from main strike and select the lowest price option"""
        try:
            logger.info(f"Finding {option_type} hedge strike from main strike: {main_strike}")
            
            if not os.path.exists(NFO_SYMBOLS_FILE):
                logger.error(f"{NFO_SYMBOLS_FILE} file not found")
                return None, None
            
            df = pd.read_csv(NFO_SYMBOLS_FILE)
            
            expiry_dt = datetime.strptime(expiry_date, '%d-%b-%Y')
            expiry_str_file = expiry_dt.strftime("%d-%b-%Y").upper()
            
            df = df[
                (df["Instrument"].str.strip() == INSTRUMENT_TYPE) & 
                (df["Symbol"].str.strip() == INDEX_SYMBOL) & 
                (df["Expiry"].str.strip().str.upper() == expiry_str_file)
            ]
            
            if df.empty:
                logger.error(f"No {INDEX_SYMBOL} options found for expiry {expiry_str_file}")
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
            
            search_strikes = [s for s in search_strikes if abs(s - main_strike) <= HEDGE_MAX_SEARCH_DISTANCE]
            
            best_strike = None
            best_ltp = float('inf')
            
            for strike in search_strikes:
                option_data = df[(df['StrikePrice'] == strike) & (df['OptionType'] == option_type)]
                if not option_data.empty:
                    symbol = option_data.iloc[0]['TradingSymbol']
                    token = option_data.iloc[0]['Token']
                    ltp = self.get_option_ltp(symbol, token)
                    
                    if HEDGE_PRICE_RANGE[0] <= ltp <= HEDGE_PRICE_RANGE[1]:
                        if ltp < best_ltp:
                            best_ltp = ltp
                            best_strike = strike
                            logger.debug(f"Found better {option_type} hedge strike: {strike} @ {ltp:.2f}")
                    else:
                        logger.debug(f"Strike {strike} {option_type}: {ltp:.2f} (outside range {HEDGE_PRICE_RANGE})")
            
            if best_strike:
                logger.info(f"Selected {option_type} hedge strike: {best_strike} @ {best_ltp:.2f} (lowest price)")
                return best_strike, best_ltp
            else:
                logger.warning(f"No {option_type} hedge strike found in price range {HEDGE_PRICE_RANGE}")
                return None, None
            
        except Exception as e:
            logger.error(f"Error finding {option_type} hedge strike: {str(e)}")
            return None, None
    
    def get_option_delta(self, strike, option_type, expiry_date):
        """Get delta for specific option - simplified version"""
        try:
            strike_float = float(strike)
            spot_price = self.get_spot_price()
            if spot_price == 0:
                logger.warning("Spot price unavailable for delta calculation, using default")
                return 0.5 if option_type == 'CE' else -0.5
                
            moneyness = (strike_float - spot_price) / spot_price
            
            if option_type == 'CE':
                if moneyness < -0.01:  # ITM call
                    return 0.7
                elif moneyness > 0.01:  # OTM call
                    return 0.3
                else:  # ATM call
                    return 0.5
            else:  # PE
                if moneyness < -0.01:  # OTM put
                    return -0.3
                elif moneyness > 0.01:  # ITM put
                    return -0.7
                else:  # ATM put
                    return -0.5
                    
        except Exception as e:
            logger.error(f"Error calculating delta: {str(e)}")
            return 0.5 if option_type == 'CE' else -0.5
    
    def get_option_symbol_from_chain(self, strike, option_type, expiry_date):
        """Get the correct trading symbol from option chain"""
        try:
            logger.debug(f"Getting symbol for {option_type} strike {strike}")
            
            if not os.path.exists(NFO_SYMBOLS_FILE):
                logger.error(f"{NFO_SYMBOLS_FILE} file not found")
                return None, None
            
            df = pd.read_csv(NFO_SYMBOLS_FILE)
            
            expiry_dt = datetime.strptime(expiry_date, '%d-%b-%Y')
            expiry_str_file = expiry_dt.strftime("%d-%b-%Y").upper()
            
            strike_float = float(strike)
            
            option_data = df[
                (df['StrikePrice'] == strike_float) & 
                (df['OptionType'] == option_type) &
                (df["Instrument"].str.strip() == INSTRUMENT_TYPE) & 
                (df["Symbol"].str.strip() == INDEX_SYMBOL) & 
                (df["Expiry"].str.strip().str.upper() == expiry_str_file)
            ]
            
            if not option_data.empty:
                symbol = option_data.iloc[0]['TradingSymbol']
                token = option_data.iloc[0]['Token']
                logger.debug(f"Found symbol: {symbol}, token: {token}")
                return symbol, token
            
            logger.warning(f"No symbol found for {option_type} strike {strike}")
            return None, None
            
        except Exception as e:
            logger.error(f"Error getting symbol from chain: {str(e)}")
            return None, None
    
    def place_straddle_order(self, strike, option_type, action, quantity):
        """Place order for straddle strategy"""
        try:
            logger.info(f"Placing {action} order for {option_type} strike {strike}, quantity: {quantity}")
            
            expiry_date = self.get_expiry_date()
            
            symbol, token = self.get_option_symbol_from_chain(strike, option_type, expiry_date)
            if not symbol:
                logger.error(f"Could not find symbol for {strike} {option_type}")
                return False
            
            client = self.client_manager.clients[0][2]
            
            buy_sell = "B" if action.upper() == "BUY" else "S"
            
            result = client.place_order(
                buy_or_sell=buy_sell, 
                product_type=ORDER_PRODUCT_TYPE, 
                exchange=ORDER_EXCHANGE,
                tradingsymbol=symbol, 
                quantity=quantity, 
                discloseqty=0,
                price_type=ORDER_PRICE_TYPE, 
                price=0, 
                trigger_price=0,
                retention=ORDER_RETENTION, 
                remarks=ORDER_REMARKS
            )
            
            if result and 'stat' in result and result['stat'] == 'Ok':
                logger.info(f"Successfully placed {action} {quantity} {symbol}")
                return True
            else:
                logger.error(f"Failed to place {action} order for {symbol}: {result}")
                return False
            
        except Exception as e:
            logger.error(f"Error placing {action} order: {str(e)}")
            return False
    
    def record_strategy_data(self, strike, ce_ltp, pe_ltp, ce_delta, pe_delta):
        """Record strategy data to CSV file in log directory"""
        try:
            file_exists = os.path.exists(self.strategy_data_file)
            
            with open(self.strategy_data_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['Timestamp', 'Strike', 'CE_LTP', 'PE_LTP', 
                                   'CE_Delta', 'PE_Delta', 'Total_Delta'])
                
                total_delta = self.calculate_portfolio_delta()
                timestamp = datetime.now(IST_TIMEZONE).strftime('%Y-%m-%d %H:%M:%S IST')
                
                writer.writerow([timestamp, strike, ce_ltp, pe_ltp, 
                               ce_delta, pe_delta, total_delta])
                
                logger.debug(f"Recorded strategy data: Strike {strike}, Total Delta: {total_delta:.2f}")
                
        except Exception as e:
            logger.error(f"Error recording strategy data: {str(e)}")
    
    def calculate_portfolio_delta(self):
        """Calculate total portfolio delta"""
        try:
            total_delta = 0
            expiry_date = self.get_expiry_date()
            for key, position in self.positions.items():
                strike, option_type = key.split('_')
                option_delta = self.get_option_delta(float(strike), option_type, expiry_date)
                total_delta += option_delta * position['quantity']
            logger.debug(f"Portfolio delta calculated: {total_delta:.2f}")
            return total_delta
        except Exception as e:
            logger.error(f"Error calculating portfolio delta: {str(e)}")
            return 0
    
    def execute_hedge(self, current_delta):
        """Execute hedging based on current delta"""
        try:
            logger.info(f"Executing hedge for delta: {current_delta:.2f}")
            
            expiry_date = self.get_expiry_date()
            
            if current_delta > DELTA_THRESHOLD:
                main_strike = list(self.positions.keys())[0].split('_')[0]
                hedge_strike, hedge_ltp = self.find_hedge_strike_directional(float(main_strike), 'PE', expiry_date)
                if hedge_strike:
                    if self.place_straddle_order(hedge_strike, 'PE', 'BUY', LOT_SIZE):
                        self.positions[f"{hedge_strike}_PE"] = {
                            'strike': hedge_strike, 'type': 'PE', 'quantity': LOT_SIZE,
                            'delta': self.get_option_delta(hedge_strike, 'PE', expiry_date)
                        }
                        logger.info(f"Hedged with PE {hedge_strike} @ {hedge_ltp:.2f}")
            
            elif current_delta < -DELTA_THRESHOLD:
                main_strike = list(self.positions.keys())[0].split('_')[0]
                hedge_strike, hedge_ltp = self.find_hedge_strike_directional(float(main_strike), 'CE', expiry_date)
                if hedge_strike:
                    if self.place_straddle_order(hedge_strike, 'CE', 'BUY', LOT_SIZE):
                        self.positions[f"{hedge_strike}_CE"] = {
                            'strike': hedge_strike, 'type': 'CE', 'quantity': LOT_SIZE,
                            'delta': self.get_option_delta(hedge_strike, 'CE', expiry_date)
                        }
                        logger.info(f"Hedged with CE {hedge_strike} @ {hedge_ltp:.2f}")
                        
        except Exception as e:
            logger.error(f"Error executing hedge: {str(e)}")
    
    def monitor_positions(self):
        """Monitor positions and execute hedging when needed"""
        logger.info("Starting position monitoring")
        
        last_delta_update = None
        
        while self.is_running:
            try:
                current_time = datetime.now(IST_TIMEZONE)
                
                current_delta = self.calculate_portfolio_delta()
                logger.debug(f"Current portfolio delta: {current_delta:.2f}")
                
                if abs(current_delta) >= DELTA_THRESHOLD:
                    logger.warning(f"Delta threshold breached: {current_delta:.2f}")
                    winsound.Beep(1000, 500)
                    self.execute_hedge(current_delta)
                
                current_minute = current_time.minute
                if (current_minute % DELTA_UPDATE_MINUTES == 0 and 
                    current_time.second < 10 and
                    (last_delta_update is None or 
                     (current_time - last_delta_update).total_seconds() >= 300)):
                    
                    if self.positions:
                        main_strike = list(self.positions.keys())[0].split('_')[0]
                        expiry_date = self.get_expiry_date()
                        ce_delta = self.get_option_delta(float(main_strike), 'CE', expiry_date)
                        pe_delta = self.get_option_delta(float(main_strike), 'PE', expiry_date)
                        self.record_strategy_data(main_strike, 0, 0, ce_delta, pe_delta)
                        last_delta_update = current_time
                        logger.info(f"Recorded delta data at {current_time.strftime('%H:%M:%S')} IST")
                
                sleep_time.sleep(60)
                
            except Exception as e:
                logger.error(f"Monitoring error: {str(e)}")
                sleep_time.sleep(60)
    
    def execute_strategy(self):
        """Main strategy execution - BUY first then SELL for leverage"""
        if self.is_running or self.strategy_executed:
            logger.warning("Strategy already running or executed today")
            return False
            
        try:
            logger.info("Starting strategy execution")
            self.is_running = True
            
            spot_price = self.get_spot_price()
            if spot_price == 0:
                logger.error("Cannot execute strategy: Spot price unavailable")
                self.is_running = False
                return False
            
            expiry_date = self.get_expiry_date()
            logger.info(f"Using expiry: {expiry_date}, Spot: {spot_price:.2f}")
            
            optimal_strike, ce_ltp, pe_ltp = self.find_optimal_straddle_strike(spot_price, expiry_date)
            if not optimal_strike:
                logger.error("No suitable straddle strike found")
                self.is_running = False
                return False
            
            logger.info(f"Optimal straddle strike: {optimal_strike}, CE: {ce_ltp:.2f}, PE: {pe_ltp:.2f}")
            
            ce_hedge_strike, ce_hedge_ltp = self.find_hedge_strike_directional(float(optimal_strike), 'CE', expiry_date)
            pe_hedge_strike, pe_hedge_ltp = self.find_hedge_strike_directional(float(optimal_strike), 'PE', expiry_date)
            
            if ce_hedge_strike:
                if not self.place_straddle_order(ce_hedge_strike, 'CE', 'BUY', LOT_SIZE):
                    logger.warning("Failed to place CE hedge BUY order")
                else:
                    self.positions[f"{ce_hedge_strike}_CE"] = {
                        'strike': ce_hedge_strike, 'type': 'CE', 'quantity': LOT_SIZE,
                        'delta': self.get_option_delta(ce_hedge_strike, 'CE', expiry_date)
                    }
            
            if pe_hedge_strike:
                if not self.place_straddle_order(pe_hedge_strike, 'PE', 'BUY', LOT_SIZE):
                    logger.warning("Failed to place PE hedge BUY order")
                else:
                    self.positions[f"{pe_hedge_strike}_PE"] = {
                        'strike': pe_hedge_strike, 'type': 'PE', 'quantity': LOT_SIZE,
                        'delta': self.get_option_delta(pe_hedge_strike, 'PE', expiry_date)
                    }
            
            if not self.place_straddle_order(optimal_strike, 'CE', 'SELL', LOT_SIZE):
                logger.error("Failed to place CE SELL order")
                self.is_running = False
                return False
            else:
                self.positions[f"{optimal_strike}_CE"] = {
                    'strike': optimal_strike, 'type': 'CE', 'quantity': -LOT_SIZE,
                    'delta': self.get_option_delta(optimal_strike, 'CE', expiry_date)
                }
            
            if not self.place_straddle_order(optimal_strike, 'PE', 'SELL', LOT_SIZE):
                logger.error("Failed to place PE SELL order")
                self.is_running = False
                return False
            else:
                self.positions[f"{optimal_strike}_PE"] = {
                    'strike': optimal_strike, 'type': 'PE', 'quantity': -LOT_SIZE,
                    'delta': self.get_option_delta(optimal_strike, 'PE', expiry_date)
                }
            
            ce_delta = self.get_option_delta(optimal_strike, 'CE', expiry_date)
            pe_delta = self.get_option_delta(optimal_strike, 'PE', expiry_date)
            self.record_strategy_data(optimal_strike, ce_ltp, pe_ltp, ce_delta, pe_delta)
            
            self.monitor_thread = threading.Thread(target=self.monitor_positions)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
            logger.info("Straddle strategy executed successfully - BUY first then SELL")
            return True
            
        except Exception as e:
            logger.error(f"Strategy execution failed: {str(e)}")
            self.is_running = False
            return False
    
    def stop_strategy(self):
        """Stop the strategy"""
        logger.info("Stopping strategy")
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Strategy stopped")

    def _update_positions_greek(self):
        """Update Greek table with position details including delta and total row"""
        try:
            current_time = datetime.now(IST_TIMEZONE)
            
            if current_time.time() < ENTRY_TIME or current_time.time() > EXIT_TIME:
                logger.debug("Outside trading hours - skipping Greek table update")
                return
                
            if current_time.minute % DELTA_UPDATE_MINUTES != 0:
                logger.debug("Not time for Greek table update yet")
                return
                
            if not hasattr(self, 'client_manager') or not self.client_manager.clients:
                logger.warning("No client manager available for Greek table update")
                return

            client_name, client_id, primary_client = self.client_manager.clients[0]
            logger.debug(f"Updating Greek table for client: {client_name}")

            try:
                positions = primary_client.get_positions()
                if positions is None:
                    logger.warning("No positions returned for Greek table")
                    return
            except Exception as e:
                self.ui.log_message(client_name, f"Error fetching positions for Greek table: {str(e)}")
                logger.error(f"Error fetching positions for Greek table: {str(e)}")
                return

            self.ui.GreekTable.setRowCount(0)
            self.ui.GreekTable.setColumnCount(6)
            
            headers = ["Position", "Entry Price", "Current Price", "Exit Price", "P&L", "Delta"]
            self.ui.GreekTable.setHorizontalHeaderLabels(headers)

            if not positions:
                logger.debug("No positions to display in Greek table")
                return

            rows_data = []
            total_pnl = 0
            total_delta = 0

            for pos in positions:
                try:
                    symbol = pos.get("tsym", "")
                    if not symbol:
                        continue

                    buy_qty = (
                        int(float(pos.get("totbuyqty") or 0))
                        or int(float(pos.get("cfbuyqty") or 0))
                        or int(float(pos.get("daybuyqty") or 0))
                    )
                    sell_qty = (
                        int(float(pos.get("totsellqty") or 0))
                        or int(float(pos.get("cfsellqty") or 0))
                        or int(float(pos.get("daysellqty") or 0))
                    )
                    net_qty = int(float(pos.get("netqty") or 0))

                    if buy_qty == 0 and sell_qty == 0:
                        continue

                    position_desc = ""
                    entry_price = 0.0
                    exit_price = 0.0
                    
                    if net_qty < 0:
                        position_desc = f"-{abs(net_qty)}x {symbol}"
                        entry_price = (
                            float(pos.get("totsellavgprc") or 0)
                            or float(pos.get("cfsellavgprc") or 0)
                            or float(pos.get("daysellavgprc") or 0)
                        )
                        exit_price = 0.0
                        
                    elif net_qty > 0:
                        position_desc = f"+{net_qty}x {symbol}"
                        entry_price = (
                            float(pos.get("totbuyavgprc") or 0)
                            or float(pos.get("cfbuyavgprc") or 0)
                            or float(pos.get("daybuyavgprc") or 0)
                        )
                        exit_price = 0.0
                        
                    else:
                        position_desc = f"0x {symbol} (Closed)"
                        if buy_qty > 0:
                            entry_price = (
                                float(pos.get("totbuyavgprc") or 0)
                                or float(pos.get("cfbuyavgprc") or 0)
                                or float(pos.get("daybuyavgprc") or 0)
                            )
                            exit_price = (
                                float(pos.get("totsellavgprc") or 0)
                                or float(pos.get("cfsellavgprc") or 0)
                                or float(pos.get("daysellavgprc") or 0)
                            )
                        else:
                            entry_price = (
                                float(pos.get("totsellavgprc") or 0)
                                or float(pos.get("cfsellavgprc") or 0)
                                or float(pos.get("daysellavgprc") or 0)
                            )
                            exit_price = (
                                float(pos.get("totbuyavgprc") or 0)
                                or float(pos.get("cfbuyavgprc") or 0)
                                or float(pos.get("daybuyavgprc") or 0)
                            )

                    current_price = float(pos.get("lp") or 0)
                    
                    mtm = float(pos.get("urmtom") or 0)
                    realized_pnl = float(pos.get("rpnl") or 0)
                    total_position_pnl = mtm + realized_pnl
                    total_pnl += total_position_pnl

                    delta = self._calculate_option_delta(symbol, current_price)
                    total_delta += delta

                    row_data = {
                        "position_desc": position_desc,
                        "entry_price": entry_price,
                        "current_price": current_price,
                        "exit_price": exit_price,
                        "pnl": total_position_pnl,
                        "delta": delta,
                        "net_qty": net_qty
                    }
                    rows_data.append(row_data)

                except Exception as e:
                    self.ui.log_message("GreekTableError", f"Error processing position {symbol}: {str(e)}")
                    logger.error(f"Error processing position {symbol}: {str(e)}")
                    continue

            for row_idx, row_data in enumerate(rows_data):
                self.ui.GreekTable.insertRow(row_idx)

                items = [
                    QTableWidgetItem(row_data["position_desc"]),
                    QTableWidgetItem(f"{row_data['entry_price']:.2f}"),
                    QTableWidgetItem(f"{row_data['current_price']:.2f}"),
                    QTableWidgetItem(f"{row_data['exit_price']:.2f}"),
                    QTableWidgetItem(f"{row_data['pnl']:.2f}"),
                    QTableWidgetItem(f"{row_data['delta']:.4f}")
                ]

                for col, item in enumerate(items):
                    item.setTextAlignment(Qt.AlignCenter)
                    
                    if col == 4:
                        pnl_value = float(item.text())
                        item.setForeground(QColor("green") if pnl_value > 0 else QColor("red") if pnl_value < 0 else QColor("black"))
                    
                    elif col == 5:
                        delta_value = float(item.text())
                        item.setForeground(QColor("green") if delta_value > 0 else QColor("red") if delta_value < 0 else QColor("black"))

                    self.ui.GreekTable.setItem(row_idx, col, item)

            if rows_data:
                self.ui.GreekTable.insertRow(len(rows_data))
                
                total_items = [
                    QTableWidgetItem("Total"),
                    QTableWidgetItem(""),
                    QTableWidgetItem(""),
                    QTableWidgetItem(""),
                    QTableWidgetItem(f"{total_pnl:.2f}"),
                    QTableWidgetItem(f"{total_delta:.4f}")
                ]
                
                for col, item in enumerate(total_items):
                    item.setTextAlignment(Qt.AlignCenter)
                    if col == 4:
                        item.setForeground(QColor("green") if total_pnl > 0 else QColor("red") if total_pnl < 0 else QColor("black"))
                        item.setFont(QFont("Arial", 10, QFont.Bold))
                    elif col == 5:
                        item.setForeground(QColor("green") if total_delta > 0 else QColor("red") if total_delta < 0 else QColor("black"))
                        item.setFont(QFont("Arial", 10, QFont.Bold))
                    
                    self.ui.GreekTable.setItem(len(rows_data), col, item)

            self.ui.GreekTable.resizeColumnsToContents()

            self.ui.log_message("GreekTable", f"Updated {len(rows_data)} positions | Total P&L: {total_pnl:.2f} | Total Delta: {total_delta:.4f} at {current_time.strftime('%H:%M:%S')}")
            logger.info(f"Greek table updated with {len(rows_data)} positions, Total P&L: {total_pnl:.2f}, Total Delta: {total_delta:.4f}")

        except Exception as e:
            self.ui.log_message("GreekTableError", f"Error updating Greek table: {str(e)}")
            logger.error(f"Error updating Greek table: {str(e)}")

    def _calculate_option_delta(self, symbol, current_price):
        """Calculate delta for an option symbol using NSE data and Black-Scholes"""
        try:
            expiry_date_str = self.get_expiry_date()
            
            if not os.path.exists(NFO_SYMBOLS_FILE):
                logger.error(f"{NFO_SYMBOLS_FILE} file not found for delta calculation")
                return 0.0
            
            df = pd.read_csv(NFO_SYMBOLS_FILE)
            
            option_data = df[df['TradingSymbol'] == symbol]
            
            if option_data.empty:
                logger.warning(f"Option {symbol} not found in {NFO_SYMBOLS_FILE} for delta calculation")
                return 0.0
            
            strike_price = float(option_data.iloc[0]['StrikePrice'])
            option_type = option_data.iloc[0]['OptionType']
            
            deltas = self._get_nse_deltas_for_strategy(expiry_date_str, strike_price, option_type)
            
            if (strike_price, option_type) in deltas:
                delta_value = deltas[(strike_price, option_type)]
                logger.debug(f"Delta for {symbol}: {delta_value:.4f}")
                return delta_value
            else:
                fallback_delta = self._calculate_fallback_delta(strike_price, option_type)
                logger.debug(f"Using fallback delta for {symbol}: {fallback_delta:.4f}")
                return fallback_delta
                    
        except Exception as e:
            logger.error(f"Error calculating delta for {symbol}: {str(e)}")
            return 0.0

    def _get_nse_deltas_for_strategy(self, expiry_date_str, target_strike=None, target_option_type=None):
        """Get delta values from NSE using Black-Scholes calculation"""
        deltas = {}
        try:
            logger.debug(f"Fetching NSE deltas for expiry: {expiry_date_str}")
            
            df_raw, S = self._fetch_option_chain(INDEX_SYMBOL)
            if S is None:
                logger.warning("Could not fetch underlying spot from NSE, using fallback delta")
                return deltas

            ce = pd.json_normalize(df_raw["CE"]).add_prefix("CE.")
            pe = pd.json_normalize(df_raw["PE"]).add_prefix("PE.")
            base = df_raw[["strikePrice", "expiryDate"]]
            df = pd.concat([base, ce, pe], axis=1)

            df = df[df["expiryDate"] == expiry_date_str].copy()
            
            if df.empty:
                logger.warning(f"No NSE data found for expiry {expiry_date_str}")
                return deltas

            if target_strike is not None and target_option_type is not None:
                df = df[df['strikePrice'] == target_strike]
                if df.empty:
                    logger.warning(f"No NSE data found for strike {target_strike}")
                    return deltas

            T = self._time_to_expiry_yrs(expiry_date_str)
            
            df["CE.iv"] = df.get("CE.impliedVolatility", np.nan) / 100.0
            df["PE.iv"] = df.get("PE.impliedVolatility", np.nan) / 100.0

            q_est = self._estimate_q_from_parity(df, S=S, r=RISK_FREE_RATE, T=T)

            for _, row in df.iterrows():
                strike = row["strikePrice"]
                
                if not pd.isna(row["CE.iv"]):
                    ce_delta = self._bs_delta(
                        S=S, K=strike, T=T, r=RISK_FREE_RATE, q=q_est, 
                        sigma=row["CE.iv"], opt_type="C"
                    )
                    deltas[(strike, 'CE')] = ce_delta
                
                if not pd.isna(row["PE.iv"]):
                    pe_delta = self._bs_delta(
                        S=S, K=strike, T=T, r=RISK_FREE_RATE, q=q_est, 
                        sigma=row["PE.iv"], opt_type="P"
                    )
                    deltas[(strike, 'PE')] = pe_delta
                    
            logger.info(f"Calculated NSE deltas for {len(deltas)} options")
            
        except Exception as e:
            logger.error(f"Failed to get NSE deltas: {str(e)}")
        
        return deltas

    def _calculate_fallback_delta(self, strike_price, option_type):
        """Fallback delta calculation when NSE data is unavailable"""
        try:
            spot_price = self.get_spot_price()
            if spot_price == 0:
                logger.warning("Spot price unavailable for fallback delta calculation")
                return 50.0 if option_type == 'CE' else -50.0

            moneyness = (strike_price - spot_price) / spot_price

            if option_type == 'CE':
                if moneyness < -0.02:
                    return 85.0
                elif moneyness < -0.01:
                    return 70.0
                elif moneyness < 0.01:
                    return 50.0
                elif moneyness < 0.02:
                    return 30.0
                else:
                    return 15.0
            else:
                if moneyness > 0.02:
                    return -85.0
                elif moneyness > 0.01:
                    return -70.0
                elif moneyness > -0.01:
                    return -50.0
                elif moneyness > -0.02:
                    return -30.0
                else:
                    return -15.0
                    
        except Exception as e:
            logger.error(f"Error in fallback delta calculation: {str(e)}")
            return 50.0 if option_type == 'CE' else -50.0

    def _fetch_option_chain(self, symbol=INDEX_SYMBOL):
        """Fetch option chain data from NSE"""
        try:
            logger.debug(f"Fetching option chain for {symbol}")
            
            s = requests.Session()
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.nseindia.com/option-chain",
            }
            s.headers.update(headers)
            
            for url in ("https://www.nseindia.com", "https://www.nseindia.com/option-chain"):
                try:
                    s.get(url, timeout=10)
                except:
                    pass
                sleep_time.sleep(0.3)
            
            url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
            resp = s.get(url, timeout=15)
            
            if resp.status_code == 200:
                data = resp.json()
                records = data.get("records", {}).get("data", [])
                underlying_value = data.get("records", {}).get("underlyingValue", None)
                logger.debug(f"Successfully fetched option chain, underlying value: {underlying_value}")
                return pd.DataFrame(records), underlying_value
            else:
                logger.warning(f"Failed to fetch option chain, status code: {resp.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to fetch NSE option chain: {str(e)}")
        
        return pd.DataFrame(), None

    def _time_to_expiry_yrs(self, expiry_str):
        """Calculate time to expiry in years"""
        try:
            now_ist = datetime.now(IST_TIMEZONE)
            expiry_dt = datetime.strptime(expiry_str, "%d-%b-%Y").replace(
                hour=15, minute=30, second=0, tzinfo=IST_TIMEZONE
            )
            secs = max((expiry_dt - now_ist).total_seconds(), 0.0)
            time_to_expiry = secs / (365.0 * 24.0 * 3600.0)
            logger.debug(f"Time to expiry: {time_to_expiry:.4f} years")
            return time_to_expiry
        except Exception as e:
            logger.warning(f"Error calculating time to expiry, using default: {str(e)}")
            return DEFAULT_TIME_TO_EXPIRY

    def _bs_delta(self, S, K, T, r, q, sigma, opt_type="C"):
        """Black-Scholes delta calculation"""
        try:
            if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
                logger.warning("Invalid parameters for Black-Scholes calculation")
                return 50.0 if opt_type == "C" else -50.0
                
            d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
            if opt_type.upper() == "C":
                delta = math.exp(-q * T) * norm.cdf(d1)
            else:
                delta = math.exp(-q * T) * (norm.cdf(d1) - 1.0)
            
            delta_percent = delta * 100
            logger.debug(f"Black-Scholes delta: S={S}, K={K}, T={T}, sigma={sigma}, delta={delta_percent:.2f}")
            return delta_percent
            
        except Exception as e:
            logger.error(f"Error in Black-Scholes calculation: {str(e)}")
            return 50.0 if opt_type == "C" else -50.0

    def _estimate_q_from_parity(self, rows, S, r, T):
        """Estimate dividend yield from put-call parity"""
        try:
            qs = []
            for _, row in rows.iterrows():
                C = row.get("CE.lastPrice", np.nan)
                P = row.get("PE.lastPrice", np.nan)
                K = row.get("strikePrice", np.nan)
                if any(pd.isna(x) for x in (C, P, K)) or S <= 0 or T <= 0:
                    continue
                rhs = (C - P + K * math.exp(-r * T)) / S
                if rhs > 0:
                    q = -math.log(rhs) / T
                    if -0.05 < q < 0.20:
                        qs.append(q)
            q_est = float(np.median(qs)) if qs else 0.0
            logger.debug(f"Estimated dividend yield: {q_est:.4f}")
            return q_est
        except Exception as e:
            logger.warning(f"Error estimating dividend yield: {str(e)}")
            return 0.0