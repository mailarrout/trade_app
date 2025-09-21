import os
import math
import pandas as pd
from datetime import datetime, time, timedelta
from PyQt5.QtCore import QTimer
from pytz import timezone
from functools import wraps

IST = timezone('Asia/Kolkata')

# ---------------------- Utility Decorators ----------------------
def safe_log(context: str):
    """Decorator to log errors gracefully without crashing."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.log("ERROR", f"{context}: {str(e)}")
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
    
    @staticmethod
    def is_trading_day() -> bool:
        """Check if current day is a trading day (Monday-Friday)"""
        return ISTTimeUtils.now().weekday() < 5

# ---------------------- Monthly Straddle Strategy ----------------------
class MonthlyStraddleStrategy:
    """
    Monthly Straddle Strategy Implementation
    
    1. Entry Timing & Instrument Selection
    - Entry Time: 9:20 AM
    - Entry Day: Next trading day after the last Thursday expiry of the current month
    - Instrument Filtering: OPTIDX, NIFTY, Next month expiry
    
    2. Strike Price Selection Process
    - Find CE and PE with similar prices (10-25 range)
    - Move ±500 points from spot to find balanced strikes
    
    3. Hedging Setup
    - Sell ATM straddle + Buy OTM protective options
    - CE side: Buy higher strike CE with LTP 10-25
    - PE side: Buy lower strike PE with LTP 10-25
    
    4. Monitoring & Adjustment
    - Define lower_range and higher_range
    - Exit when spot approaches within 20-30 points of either range
    - Re-enter with fresh positions
    
    5. Continuous Monitoring
    - Maintain straddle structure with proper hedging
    """
    
    def __init__(self, ui, client_manager):
        self.ui = ui
        self.client_manager = client_manager
        self.spot_price = None
        self.lower_range = None
        self.higher_range = None
        
        self.log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.current_state_file = os.path.join(
            self.log_dir, f"{ISTTimeUtils.current_date_str()}_monthly_straddle_state.csv"
        )
        
        # Strategy state
        self.state = "WAITING"  # WAITING | ACTIVE | ADJUSTING | COMPLETED
        self.positions = {
            'ce_sell': None,  # Main CE sell position
            'pe_sell': None,  # Main PE sell position
            'ce_buy': None,   # Protective CE buy position
            'pe_buy': None    # Protective PE buy position
        }
        
        # Setup timer for strategy execution
        self.strategy_timer = QTimer()
        self.strategy_timer.timeout.connect(self._check_and_execute_strategy)
        self.strategy_timer.start(30000)  # Check every 30 seconds
        
        logger.log("MONTHLY STRADDLE", "Strategy initialized - Runs at 9:20 AM on entry days")
        
        # Connect UI button if needed
        if hasattr(self.ui, 'ExecuteMonthlyStraddleQPushButton'):
            self.ui.ExecuteMonthlyStraddleQPushButton.clicked.connect(self.on_execute_strategy_clicked)

    # ---------------------- Strategy Execution ----------------------
    def on_execute_strategy_clicked(self):
        """Manual execution of the strategy"""
        current_time = ISTTimeUtils.current_time()
        
        if not self._is_entry_day():
            logger.log("MONTHLY STRADDLE", "Not an entry day - waiting for next month's expiry Thursday")
            return
            
        if current_time < time(9, 15) or current_time > time(15, 15):
            logger.log("MONTHLY STRADDLE", "Outside trading hours")
            return
            
        logger.log("MONTHLY STRADDLE", "Manually executing Monthly Straddle strategy")
        self._run_strategy_cycle()

    def _check_and_execute_strategy(self):
        """Check if it's time to execute the strategy"""
        current_time = ISTTimeUtils.current_time()
        
        # Check if it's an entry day and time
        if (self._is_entry_day() and 
            current_time.hour == 9 and current_time.minute == 20 and
            self.state == "WAITING"):
            self._run_strategy_cycle()
            
        # Monitor positions during market hours
        if (time(9, 30) <= current_time <= time(15, 15) and 
            self.state == "ACTIVE"):
            self._monitor_positions()

    def _run_strategy_cycle(self):
        """Main strategy execution cycle"""
        logger.log("MONTHLY STRADDLE", "=== Running Monthly Straddle Strategy ===")
        
        # Get current spot price
        if not self._get_spot_price():
            logger.log("ERROR", "Failed to get spot price")
            return
            
        # Get appropriate expiry date
        expiry_date = self._get_expiry_date()
        if not expiry_date:
            logger.log("ERROR", "Failed to get expiry date")
            return
            
        # Select strike prices
        strikes = self._select_strikes(expiry_date)
        if not strikes:
            logger.log("ERROR", "Failed to select strikes")
            return
            
        # Place orders
        if self._place_orders(expiry_date, strikes):
            self.state = "ACTIVE"
            logger.log("MONTHLY STRADDLE", "Positions placed successfully")
            self._log_state("ACTIVE", "Positions placed")
        else:
            logger.log("ERROR", "Failed to place orders")

    # ---------------------- Market Data ----------------------
    @safe_log("Spot price fetch failed")
    def _get_spot_price(self) -> bool:
        """Get current NIFTY spot price"""
        try:
            if not os.path.exists("NFO_symbols.txt"):
                logger.log("ERROR", "NFO_symbols.txt not found")
                return False
                
            df = pd.read_csv("NFO_symbols.txt")
            df = df[(df["Instrument"].str.strip() == "FUTIDX") & 
                   (df["Symbol"].str.strip() == "NIFTY")]
            
            if df.empty:
                logger.log("ERROR", "No futures data found for NIFTY")
                return False
                
            token = str(df.iloc[0]['Token'])
            if not self._validate_clients():
                logger.log("ERROR", "No active client available")
                return False
                
            client = self.client_manager.clients[0][2]
            quote = client.get_quotes('NFO', token)
            
            if not quote or quote.get('stat') != 'Ok':
                logger.log("ERROR", "Failed to get quote")
                return False
                
            self.spot_price = float(quote.get('lp', 0))
            logger.log("MARKET", f"NIFTY Spot Price: {self.spot_price:.2f}")
            return True
            
        except Exception as e:
            logger.log("ERROR", f"Error getting spot price: {str(e)}")
            return False

    # ---------------------- Expiry Date Selection ----------------------
    def _is_entry_day(self) -> bool:
        """
        Check if today is the next trading day after the last Thursday expiry of current month
        """
        try:
            # Get current date
            today = ISTTimeUtils.now().date()
            
            # Load expiry dates from NFO_symbols.txt
            if not os.path.exists("NFO_symbols.txt"):
                logger.log("ERROR", "NFO_symbols.txt not found")
                return False
                
            df = pd.read_csv("NFO_symbols.txt")
            df = df[df["Instrument"].str.strip() == "OPTIDX"]
            df['Expiry'] = pd.to_datetime(df['Expiry'], format='%d-%b-%Y')
            
            # Get current month's last Thursday expiry
            current_month = today.replace(day=1)
            next_month = (current_month + timedelta(days=32)).replace(day=1)
            
            # Filter expiries for current month
            current_month_expiries = df[df['Expiry'].dt.to_period('M') == pd.Timestamp(current_month).to_period('M')]
            
            if current_month_expiries.empty:
                logger.log("WARNING", "No current month expiries found")
                return False
                
            # Find last Thursday of current month
            last_thursday = None
            for expiry in current_month_expiries['Expiry'].unique():
                expiry_date = expiry.date()
                if expiry_date.weekday() == 3:  # Thursday
                    if last_thursday is None or expiry_date > last_thursday:
                        last_thursday = expiry_date
            
            if not last_thursday:
                logger.log("WARNING", "No Thursday expiry found for current month")
                return False
                
            # Check if today is the next trading day after last Thursday
            next_trading_day = last_thursday + timedelta(days=1)
            while next_trading_day.weekday() >= 5:  # Skip weekends
                next_trading_day += timedelta(days=1)
                
            return today == next_trading_day
            
        except Exception as e:
            logger.log("ERROR", f"Error checking entry day: {str(e)}")
            return False

    def _get_expiry_date(self):
        """
        Get expiry date for next month (or current month if before 15th)
        """
        try:
            today = ISTTimeUtils.now().date()
            
            # Load expiry dates
            if not os.path.exists("NFO_symbols.txt"):
                logger.log("ERROR", "NFO_symbols.txt not found")
                return None
                
            df = pd.read_csv("NFO_symbols.txt")
            df = df[df["Instrument"].str.strip() == "OPTIDX"]
            df['Expiry'] = pd.to_datetime(df['Expiry'], format='%d-%b-%Y')
            
            # Determine which month to use
            if today.day < 15:
                # Use current month expiry
                target_month = today.replace(day=1)
            else:
                # Use next month expiry
                target_month = (today.replace(day=1) + timedelta(days=32)).replace(day=1)
            
            # Filter expiries for target month
            month_expiries = df[df['Expiry'].dt.to_period('M') == pd.Timestamp(target_month).to_period('M')]
            
            if month_expiries.empty:
                logger.log("ERROR", f"No expiries found for target month {target_month}")
                return None
                
            # Return the latest expiry date
            return month_expiries['Expiry'].max().date()
            
        except Exception as e:
            logger.log("ERROR", f"Error getting expiry date: {str(e)}")
            return None

    # ---------------------- Strike Selection ----------------------
    def _select_strikes(self, expiry_date):
        """
        Select strikes for the straddle:
        1. Find ATM strike with balanced CE/PE prices (10-25 range)
        2. Find protective strikes with LTP 10-25
        """
        try:
            # Load options data
            if not os.path.exists("NFO_symbols.txt"):
                logger.log("ERROR", "NFO_symbols.txt not found")
                return None
                
            df = pd.read_csv("NFO_symbols.txt")
            df = df[(df["Instrument"].str.strip() == "OPTIDX") & 
                   (df["Symbol"].str.strip() == "NIFTY")]
            
            # Convert expiry to match format
            expiry_str = expiry_date.strftime('%d-%b-%Y').upper()
            df['Expiry'] = df['Expiry'].str.strip().str.upper()
            
            # Filter for target expiry
            expiry_options = df[df['Expiry'] == expiry_str]
            
            if expiry_options.empty:
                logger.log("ERROR", f"No options found for expiry {expiry_str}")
                return None
                
            # Get client for LTP queries
            if not self._validate_clients():
                logger.log("ERROR", "No active client available")
                return None
                
            client = self.client_manager.clients[0][2]
            
            # Step 1: Find ATM strike with balanced CE/PE prices (10-25 range)
            atm_strike = None
            min_price_diff = float('inf')
            
            # Check strikes ±500 points from spot
            min_strike = self.spot_price - 500
            max_strike = self.spot_price + 500
            
            for strike in expiry_options['StrikePrice'].unique():
                if strike < min_strike or strike > max_strike:
                    continue
                    
                # Get CE and PE for this strike
                ce_data = expiry_options[(expiry_options['StrikePrice'] == strike) & 
                                       (expiry_options['OptionType'] == 'CE')]
                pe_data = expiry_options[(expiry_options['StrikePrice'] == strike) & 
                                       (expiry_options['OptionType'] == 'PE')]
                
                if ce_data.empty or pe_data.empty:
                    continue
                    
                # Get LTPs
                ce_ltp = self._get_ltp(client, ce_data.iloc[0]['Token'])
                pe_ltp = self._get_ltp(client, pe_data.iloc[0]['Token'])
                
                if not (10 <= ce_ltp <= 25 and 10 <= pe_ltp <= 25):
                    continue
                    
                # Check if this is the most balanced pair
                price_diff = abs(ce_ltp - pe_ltp)
                if price_diff < min_price_diff:
                    min_price_diff = price_diff
                    atm_strike = strike
                    ce_symbol = ce_data.iloc[0]['TradingSymbol']
                    pe_symbol = pe_data.iloc[0]['TradingSymbol']
                    ce_token = ce_data.iloc[0]['Token']
                    pe_token = pe_data.iloc[0]['Token']
            
            if not atm_strike:
                logger.log("ERROR", "No suitable ATM strike found with balanced prices")
                return None
                
            logger.log("MONTHLY STRADDLE", f"Selected ATM strike: {atm_strike}, CE: {ce_ltp:.2f}, PE: {pe_ltp:.2f}")
            
            # Step 2: Find protective strikes
            # For CE: Buy higher strike CE with LTP 10-25
            ce_protective_strike = None
            for strike in sorted(expiry_options['StrikePrice'].unique(), reverse=True):
                if strike <= atm_strike:
                    continue
                    
                ce_data = expiry_options[(expiry_options['StrikePrice'] == strike) & 
                                       (expiry_options['OptionType'] == 'CE')]
                if ce_data.empty:
                    continue
                    
                ce_ltp_protective = self._get_ltp(client, ce_data.iloc[0]['Token'])
                if 10 <= ce_ltp_protective <= 25:
                    ce_protective_strike = strike
                    ce_protective_symbol = ce_data.iloc[0]['TradingSymbol']
                    ce_protective_token = ce_data.iloc[0]['Token']
                    break
            
            # For PE: Buy lower strike PE with LTP 10-25
            pe_protective_strike = None
            for strike in sorted(expiry_options['StrikePrice'].unique()):
                if strike >= atm_strike:
                    continue
                    
                pe_data = expiry_options[(expiry_options['StrikePrice'] == strike) & 
                                       (expiry_options['OptionType'] == 'PE')]
                if pe_data.empty:
                    continue
                    
                pe_ltp_protective = self._get_ltp(client, pe_data.iloc[0]['Token'])
                if 10 <= pe_ltp_protective <= 25:
                    pe_protective_strike = strike
                    pe_protective_symbol = pe_data.iloc[0]['TradingSymbol']
                    pe_protective_token = pe_data.iloc[0]['Token']
                    break
            
            if not ce_protective_strike or not pe_protective_strike:
                logger.log("ERROR", "Could not find suitable protective strikes")
                return None
                
            logger.log("MONTHLY STRADDLE", f"Protective CE strike: {ce_protective_strike}, PE strike: {pe_protective_strike}")
            
            return {
                'atm_strike': atm_strike,
                'ce_sell': {'symbol': ce_symbol, 'token': ce_token},
                'pe_sell': {'symbol': pe_symbol, 'token': pe_token},
                'ce_protective_strike': ce_protective_strike,
                'ce_buy': {'symbol': ce_protective_symbol, 'token': ce_protective_token},
                'pe_protective_strike': pe_protective_strike,
                'pe_buy': {'symbol': pe_protective_symbol, 'token': pe_protective_token}
            }
            
        except Exception as e:
            logger.log("ERROR", f"Error selecting strikes: {str(e)}")
            return None

    # ---------------------- Order Placement ----------------------
    def _place_orders(self, expiry_date, strikes):
        """Place all orders for the straddle strategy"""
        try:
            if not self._validate_clients():
                logger.log("ERROR", "No active client available")
                return False
                
            client = self.client_manager.clients[0][2]
            quantity = 75  # Standard NIFTY lot size
            
            # Place sell orders (ATM straddle)
            ce_sell_success = self._place_order(
                client, strikes['ce_sell']['symbol'], strikes['ce_sell']['token'], 
                'SELL', quantity, "ATM CE Sell"
            )
            
            pe_sell_success = self._place_order(
                client, strikes['pe_sell']['symbol'], strikes['pe_sell']['token'], 
                'SELL', quantity, "ATM PE Sell"
            )
            
            # Place buy orders (protective)
            ce_buy_success = self._place_order(
                client, strikes['ce_buy']['symbol'], strikes['ce_buy']['token'], 
                'BUY', quantity, "Protective CE Buy"
            )
            
            pe_buy_success = self._place_order(
                client, strikes['pe_buy']['symbol'], strikes['pe_buy']['token'], 
                'BUY', quantity, "Protective PE Buy"
            )
            
            if ce_sell_success and pe_sell_success and ce_buy_success and pe_buy_success:
                # Update positions
                self.positions = {
                    'ce_sell': strikes['ce_sell'],
                    'pe_sell': strikes['pe_sell'],
                    'ce_buy': strikes['ce_buy'],
                    'pe_buy': strikes['pe_buy']
                }
                
                # Set ranges for monitoring
                self.lower_range = self.spot_price - 500
                self.higher_range = self.spot_price + 500
                
                logger.log("MONTHLY STRADDLE", 
                          f"Ranges set - Lower: {self.lower_range}, Higher: {self.higher_range}")
                
                return True
            else:
                # Cancel any successful orders if others failed
                if ce_sell_success:
                    self._place_order(client, strikes['ce_sell']['symbol'], strikes['ce_sell']['token'], 
                                    'BUY', quantity, "Cancel CE Sell")
                if pe_sell_success:
                    self._place_order(client, strikes['pe_sell']['symbol'], strikes['pe_sell']['token'], 
                                    'BUY', quantity, "Cancel PE Sell")
                if ce_buy_success:
                    self._place_order(client, strikes['ce_buy']['symbol'], strikes['ce_buy']['token'], 
                                    'SELL', quantity, "Cancel CE Buy")
                if pe_buy_success:
                    self._place_order(client, strikes['pe_buy']['symbol'], strikes['pe_buy']['token'], 
                                    'SELL', quantity, "Cancel PE Buy")
                
                return False
                
        except Exception as e:
            logger.log("ERROR", f"Error placing orders: {str(e)}")
            return False

    def _place_order(self, client, symbol, token, action, quantity, remark):
        """Place a single order"""
        try:
            client.place_order(
                buy_or_sell=action[0].upper(),  # 'B' or 'S'
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
            logger.log("ORDER", f"{action} order placed for {symbol}")
            return True
        except Exception as e:
            logger.log("ERROR", f"Failed to place {action} order for {symbol}: {str(e)}")
            return False

    # ---------------------- Monitoring ----------------------
    def _monitor_positions(self):
        """Monitor positions and adjust if needed"""
        try:
            # Get current spot price
            if not self._get_spot_price():
                return
                
            # Check if spot is approaching range boundaries
            lower_threshold = self.lower_range + 30
            higher_threshold = self.higher_range - 30
            
            if self.spot_price <= lower_threshold or self.spot_price >= higher_threshold:
                logger.log("MONTHLY STRADDLE", 
                          f"Spot price {self.spot_price} approaching range boundary. Adjusting positions.")
                self._adjust_positions()
                
        except Exception as e:
            logger.log("ERROR", f"Error monitoring positions: {str(e)}")

    def _adjust_positions(self):
        """Adjust positions by exiting and re-entering"""
        try:
            if not self._validate_clients():
                logger.log("ERROR", "No active client available")
                return
                
            client = self.client_manager.clients[0][2]
            quantity = 75
            
            # Exit all positions
            for position_type, position in self.positions.items():
                if position:
                    action = 'BUY' if 'sell' in position_type else 'SELL'
                    self._place_order(client, position['symbol'], position['token'], 
                                    action, quantity, f"Exit {position_type}")
            
            # Wait a moment for orders to execute
            QTimer.singleShot(5000, self._reenter_positions)
            
        except Exception as e:
            logger.log("ERROR", f"Error adjusting positions: {str(e)}")

    def _reenter_positions(self):
        """Re-enter positions with new strikes"""
        try:
            # Get new spot price
            if not self._get_spot_price():
                return
                
            # Get appropriate expiry date
            expiry_date = self._get_expiry_date()
            if not expiry_date:
                return
                
            # Select new strikes based on current spot
            strikes = self._select_strikes(expiry_date)
            if not strikes:
                return
                
            # Place new orders
            if self._place_orders(expiry_date, strikes):
                logger.log("MONTHLY STRADDLE", "Positions adjusted successfully")
                self._log_state("ACTIVE", "Positions adjusted")
            else:
                logger.log("ERROR", "Failed to adjust positions")
                
        except Exception as e:
            logger.log("ERROR", f"Error reentering positions: {str(e)}")

    # ---------------------- Helper Methods ----------------------
    def _get_ltp(self, client, token):
        """Get LTP for a token"""
        try:
            quote = client.get_quotes('NFO', str(token))
            if quote and quote.get('stat') == 'Ok' and 'lp' in quote:
                return float(quote['lp'])
            else:
                logger.log("WARNING", f"Invalid quote response for token {token}")
                return 0
        except Exception as e:
            logger.log("ERROR", f"Error getting LTP for token {token}: {str(e)}")
            return 0

    def _validate_clients(self):
        return hasattr(self.ui, 'client_manager') and bool(self.ui.client_manager.clients)

    def _log_state(self, status, comments):
        """Log strategy state to CSV"""
        try:
            state_data = {
                'timestamp': ISTTimeUtils.now().strftime("%Y-%m-%d %H:%M:%S"),
                'status': status,
                'spot_price': self.spot_price,
                'lower_range': self.lower_range,
                'higher_range': self.higher_range,
                'ce_sell_symbol': self.positions['ce_sell']['symbol'] if self.positions['ce_sell'] else None,
                'pe_sell_symbol': self.positions['pe_sell']['symbol'] if self.positions['pe_sell'] else None,
                'ce_buy_symbol': self.positions['ce_buy']['symbol'] if self.positions['ce_buy'] else None,
                'pe_buy_symbol': self.positions['pe_buy']['symbol'] if self.positions['pe_buy'] else None,
                'comments': comments
            }
            
            file_exists = os.path.exists(self.current_state_file)
            pd.DataFrame([state_data]).to_csv(self.current_state_file, mode='a', 
                                            header=not file_exists, index=False)
            
        except Exception as e:
            logger.log("ERROR", f"Failed to log state: {str(e)}")