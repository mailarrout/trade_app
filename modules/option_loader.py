import os
import time
import pandas as pd
from PyQt5.QtCore import QObject, pyqtSignal, Qt
from PyQt5.QtWidgets import QPushButton, QMessageBox, QTableView, QApplication
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QColor, QFont
from datetime import datetime
from pytz import timezone
import logging

# Get logger for this module
logger = logging.getLogger(__name__)

class OptionLoader(QObject):
    spot_updated = pyqtSignal(dict)
    chain_loaded = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, ui):
        super().__init__()
        self.ui = ui
        self.IST = timezone('Asia/Kolkata')
        self.spot_price = None
        self.spot_valid = False
        self.symbol_to_token = {}
        self.position_symbols = set()  # Store position symbols for highlighting
        self._initialize_table()
        self.ui.LoadOptionsPushButton.clicked.connect(self.load_option_chain)
        logger.info(f"OptionLoader initialized at {datetime.now(self.IST).strftime('%Y-%m-%d %H:%M:%S IST')}")

    def _initialize_table(self):
        try:
            logger.debug("Initializing option table view settings")
            self.ui.OptionQTableView.setSelectionBehavior(QTableView.SelectRows)
            self.ui.OptionQTableView.setEditTriggers(QTableView.NoEditTriggers)
            self.ui.OptionQTableView.verticalHeader().setVisible(False)
            logger.debug("Option table view initialized successfully")
        except Exception as e:
            logger.error(f"Table initialization failed: {str(e)}")
            raise

    def load_option_chain(self):
        """Load option chain and update LastUpdatedQLabel"""
        start_time = time.time()
        try:
            logger.info("Starting option chain loading process")
            
            if not self._validate_clients():
                logger.warning("No valid clients found for option chain loading")
                return

            # Get positions before loading option chain
            self._get_position_symbols()
            self._update_spot_price()
            elapsed = time.time() - start_time
            timestamp = datetime.now(self.IST).strftime('%H:%M:%S IST')
            self.ui.OptionDataUpdateQLabel.setText(f"Option Updated at: {timestamp} ({elapsed:.2f}s)")
            logger.info(f"Option chain loading completed in {elapsed:.2f} seconds")
            
        except Exception as e:
            error_msg = f"Option chain load failed: {str(e)}"
            self.error_occurred.emit(str(e))
            logger.error(error_msg)

    def _get_position_symbols(self):
        """Fetch all position symbols with non-zero net quantity"""
        try:
            logger.debug("Fetching position symbols")
            
            if not self._validate_clients():
                self.position_symbols = set()
                logger.warning("No clients available for position symbol fetching")
                return

            client = self.ui.client_manager.clients[0][2]
            
            # Try different possible methods to get positions
            positions = None
            position_methods = ['get_positions', 'positions', 'get_holdings', 'holdings']
            
            for method_name in position_methods:
                if hasattr(client, method_name):
                    try:
                        logger.debug(f"Trying position method: {method_name}")
                        positions = getattr(client, method_name)()
                        logger.debug(f"Using {method_name}() to fetch positions")
                        break
                    except Exception as e:
                        logger.warning(f"Method {method_name}() failed: {str(e)}")
                        continue
            
            self.position_symbols = set()
            
            if positions:
                logger.debug(f"Processing positions data: {type(positions)}")
                
                if isinstance(positions, dict):
                    # Handle dictionary response
                    if 'data' in positions:
                        positions = positions['data']
                        logger.debug("Extracted positions from 'data' key")
                    elif 'result' in positions:
                        positions = positions['result']
                        logger.debug("Extracted positions from 'result' key")
                
                if isinstance(positions, list):
                    position_count = 0
                    for position in positions:
                        symbol = None
                        net_qty = 0
                        
                        if isinstance(position, dict):
                            # Try different possible field names for symbol
                            symbol_fields = ['tsym', 'tradingsymbol', 'symbol', 'TradingSymbol', 'TrdSym']
                            for field in symbol_fields:
                                if field in position:
                                    symbol = position[field]
                                    break
                            
                            # Try different possible field names for net quantity
                            qty_fields = ['netqty', 'netQty', 'net_quantity', 'quantity', 'netqty', 'NetQty']
                            for field in qty_fields:
                                if field in position:
                                    try:
                                        net_qty = float(position[field])
                                        break
                                    except (ValueError, TypeError):
                                        continue
                        
                        elif hasattr(position, 'tsym'):
                            symbol = position.tsym
                            if hasattr(position, 'netqty'):
                                net_qty = float(position.netqty)
                        elif hasattr(position, 'tradingsymbol'):
                            symbol = position.tradingsymbol
                            if hasattr(position, 'netqty'):
                                net_qty = float(position.netqty)
                        
                        if symbol and net_qty != 0:  # Only add if net quantity is non-zero
                            symbol_str = str(symbol).strip()
                            self.position_symbols.add(symbol_str)
                            position_count += 1
                            logger.debug(f"Position found: {symbol_str} with netqty {net_qty}")
            
            logger.info(f"Found {len(self.position_symbols)} positions with non-zero quantity")
            
        except Exception as e:
            logger.error(f"Failed to fetch positions: {str(e)}")
            self.position_symbols = set()

    def _update_spot_price(self):
        """Fetch spot price and trigger option chain load"""
        try:
            logger.info("Updating spot price")
            
            if not os.path.exists("NFO_symbols.txt"):
                error_msg = "NFO_symbols.txt not found"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            df = pd.read_csv("NFO_symbols.txt")
            logger.debug(f"Loaded NFO_symbols.txt with {len(df)} rows")
            
            # Get expiry dates from dropdown
            expiry_texts = [self.ui.ExpiryListDropDown.itemText(i) for i in range(self.ui.ExpiryListDropDown.count())]
            expiry_dates = pd.to_datetime(expiry_texts, format="%d-%b-%Y", errors="coerce")

            today = pd.to_datetime("today")
            current_month_expiries = [d for d in expiry_dates if d.month == today.month and d.year == today.year]

            # Choose last expiry of current month, else fallback
            if current_month_expiries:
                monthly_expiry = max(current_month_expiries).strftime("%d-%b-%Y").upper()
            else:
                monthly_expiry = self.ui.ExpiryListDropDown.currentText().upper()

            # --- Normalize df['Expiry'] safely ---
            # Parse with explicit format to avoid warnings
            df["Expiry_dt"] = pd.to_datetime(df["Expiry"], format="%d-%b-%Y", errors="coerce")

            # Convert to normalized string
            df["Expiry_str"] = df["Expiry_dt"].dt.strftime("%d-%b-%Y").str.upper()

            # Logging
            logger.info(f"All expiries: {expiry_texts}")
            logger.info(f"Current month expiries: {current_month_expiries}")
            logger.info(f"Monthly expiry resolved to: {monthly_expiry}")

            # Filter for FUTIDX NIFTY
            df_fut = df[
                (df["Instrument"].str.strip() == "FUTIDX") &
                (df["Symbol"].str.strip() == "NIFTY") &
                (df["Expiry_str"] == monthly_expiry)
            ]

            if not df_fut.empty:
                token = df_fut.iloc[0]["Token"]
                logger.info(f"Monthly FUTIDX Token: {token}")
            else:
                logger.error(f"No futures data found for expiry: {monthly_expiry}")

            token = str(df_fut.iloc[0]['Token'])
            logger.debug(f"Using token {token} for spot price")
            
            client = self.ui.client_manager.clients[0][2]
            quote = client.get_quotes('NFO', token)

            if not quote or quote.get('stat') != 'Ok':
                error_msg = "Failed to get quote from API"
                logger.error(error_msg)
                raise ValueError(error_msg)

            spot_price = float(quote.get('lp', 0))
            self.spot_price = {
                'raw': spot_price,
                'rounded': round(spot_price / 50) * 50,
                'timestamp': datetime.now(self.IST).strftime('%H:%M:%S IST')
            }
            self.spot_valid = True

            self.ui.SpotPriceQLabel.setText(
                f"Raw: {spot_price:.2f} | {self.spot_price['rounded']} (Updated at {self.spot_price['timestamp']})"
            )

            logger.info(f"Spot price updated: {spot_price:.2f}, Rounded: {self.spot_price['rounded']}")
            self._load_option_chain_data()

        except Exception as e:
            error_time = datetime.now(self.IST).strftime('%H:%M:%S IST')
            error_msg = f"Spot price error at {error_time}: {str(e)}"
            self.error_occurred.emit(error_msg)
            self.ui.SpotPriceQLabel.setText(f"Error getting spot price at {error_time}")
            self.ui.SpotPriceQLabel.setStyleSheet("color: red;")
            logger.error(f"Spot price update failed: {str(e)}")

    def _load_option_chain_data(self):
        """Load option chain from CSV"""
        try:
            logger.info("Loading option chain data")
            
            if not self.spot_valid:
                error_msg = "Invalid spot price - cannot load option chain"
                logger.error(error_msg)
                raise ValueError(error_msg)

            strike_count = int(self.ui.StrikeNumberQLine.text())
            expiry_date_str = self.ui.ExpiryListDropDown.currentText()
            expiry_date = pd.to_datetime(expiry_date_str, format='%d-%b-%Y')
            
            logger.debug(f"Strike count: {strike_count}, Expiry: {expiry_date_str}")

            df = pd.read_csv("NFO_symbols.txt")
            df['Expiry'] = pd.to_datetime(df['Expiry'].astype(str).str.strip(), format='%d-%b-%Y', errors='coerce')
            df_opt = df[(df["Instrument"].str.strip() == "OPTIDX") &
                        (df["Symbol"].str.strip() == "NIFTY") &
                        (df['Expiry'] == expiry_date)]

            if df_opt.empty:
                error_msg = f"No option data found for NIFTY with expiry {expiry_date_str}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            min_strike = self.spot_price['rounded'] - (strike_count * 50)
            max_strike = self.spot_price['rounded'] + (strike_count * 50)
            df_opt = df_opt[(df_opt['StrikePrice'] >= min_strike) &
                            (df_opt['StrikePrice'] <= max_strike)]
            
            logger.debug(f"Filtered strikes between {min_strike} and {max_strike}: {len(df_opt)} rows")

            # Keep only strikes with both CE and PE
            strikes_with_both = [s for s in df_opt['StrikePrice'].unique()
                                 if not df_opt[(df_opt['StrikePrice'] == s) & (df_opt['OptionType'] == 'CE')].empty and
                                    not df_opt[(df_opt['StrikePrice'] == s) & (df_opt['OptionType'] == 'PE')].empty]

            df_opt = df_opt[df_opt['StrikePrice'].isin(strikes_with_both)]
            logger.debug(f"Strikes with both CE and PE: {len(strikes_with_both)}")
            
            model = self._create_option_model(df_opt)
            self.ui.OptionQTableView.setModel(model)
            self.chain_loaded.emit(model)
            
            logger.info("Option chain data loaded successfully")

        except Exception as e:
            error_time = datetime.now(self.IST).strftime('%H:%M:%S IST')
            error_msg = f"Option chain error at {error_time}: {str(e)}"
            self.error_occurred.emit(error_msg)
            logger.error(f"Option chain load failed: {str(e)}")

    def _create_option_model(self, df):
        """Create option chain table with position highlighting"""
        try:
            logger.info("Creating option chain table model")
            
            model = QStandardItemModel()
            headers = ["CE Symbol", "CE LTP", "CE Buy", "CE Sell",
                    "Strike", "PE Symbol", "PE LTP", "PE Buy", "PE Sell"]
            widths = [200, 60, 80, 80, 60, 200, 60, 80, 80]

            model.setColumnCount(len(headers))
            model.setHorizontalHeaderLabels(headers)

            # Set model first before adding buttons
            self.ui.OptionQTableView.setModel(model)
            for i, width in enumerate(widths):
                self.ui.OptionQTableView.setColumnWidth(i, width)

            self.symbol_to_token = {}
            client = self.ui.client_manager.clients[0][2]
            
            strikes = sorted(df['StrikePrice'].unique())
            logger.debug(f"Processing {len(strikes)} unique strikes")

            for row, strike in enumerate(strikes):
                ce_rows = df[(df['StrikePrice'] == strike) & (df['OptionType'] == 'CE')]
                pe_rows = df[(df['StrikePrice'] == strike) & (df['OptionType'] == 'PE')]
                if ce_rows.empty or pe_rows.empty:
                    logger.debug(f"Skipping strike {strike}: missing CE or PE data")
                    continue

                ce_data, pe_data = ce_rows.iloc[0], pe_rows.iloc[0]
                ce_symbol = ce_data['TradingSymbol']
                pe_symbol = pe_data['TradingSymbol']
                
                self.symbol_to_token[ce_symbol] = str(ce_data['Token'])
                self.symbol_to_token[pe_symbol] = str(pe_data['Token'])
                
                ce_ltp = self._get_quote(client, ce_data['Token'])
                pe_ltp = self._get_quote(client, pe_data['Token'])

                items = [QStandardItem() for _ in range(len(headers))]
                for item in items:
                    item.setTextAlignment(Qt.AlignCenter)

                # Fill table
                items[0].setText(ce_symbol)
                items[1].setText(f"{ce_ltp:.2f}")
                items[4].setText(str(strike))
                items[5].setText(pe_symbol)
                items[6].setText(f"{pe_ltp:.2f}")

                # Highlight only symbol + LTP if in positions
                position_font = QFont()
                position_font.setBold(True)
                position_color = QColor("#FFD700")  # Gold color

                ce_in_position = ce_symbol in self.position_symbols
                pe_in_position = pe_symbol in self.position_symbols

                if ce_in_position:
                    items[0].setBackground(position_color)   # CE Symbol
                    items[1].setBackground(position_color)   # CE LTP
                    items[0].setFont(position_font)
                    items[1].setFont(position_font)

                if pe_in_position:
                    items[5].setBackground(position_color)   # PE Symbol
                    items[6].setBackground(position_color)   # PE LTP
                    items[5].setFont(position_font)
                    items[6].setFont(position_font)

                if ce_in_position or pe_in_position:
                    logger.debug(f"Highlighting position symbols: CE={ce_in_position}, PE={pe_in_position} for strike {strike}")

                # Place items in model
                for col, item in enumerate(items):
                    model.setItem(row, col, item)

                # Buttons
                ce_buy = self._create_action_button("BUY", ce_symbol, "#0a631c", "#3C8155")
                ce_sell = self._create_action_button("SELL", ce_symbol, "#672108", "#A25353")
                pe_buy = self._create_action_button("BUY", pe_symbol, "#0a631c", "#3C8155")
                pe_sell = self._create_action_button("SELL", pe_symbol, "#672108", "#A25353")

                self.ui.OptionQTableView.setIndexWidget(model.index(row, 2), ce_buy)
                self.ui.OptionQTableView.setIndexWidget(model.index(row, 3), ce_sell)
                self.ui.OptionQTableView.setIndexWidget(model.index(row, 7), pe_buy)
                self.ui.OptionQTableView.setIndexWidget(model.index(row, 8), pe_sell)

                # Highlight spot row (only if not already highlighted as position)
                if strike == self.spot_price['rounded'] and not (ce_in_position or pe_in_position):
                    for col in range(model.columnCount()):
                        model.item(row, col).setBackground(QColor("#570B4F"))

            self.ui.OptionQTableView.resizeColumnsToContents()
            self.ui.OptionQTableView.viewport().update()
            QApplication.processEvents()
            
            # Log highlighting summary
            position_count = len([s for s in self.position_symbols if s in self.symbol_to_token])
            logger.info(f"Position highlighting complete: {position_count} positions highlighted out of {len(self.position_symbols)} total positions")
            
            return model

        except Exception as e:
            logger.error(f"Error creating option model: {str(e)}")
            raise

    def _get_quote(self, client, token):
        try:
            quote = client.get_quotes('NFO', str(token))
            if quote and quote.get('stat') == 'Ok':
                return float(quote.get('lp', 0))
            else:
                logger.warning(f"Failed to get quote for token {token}")
                return 0
        except Exception as e:
            logger.error(f"Error getting quote for token {token}: {str(e)}")
            return 0

    def _validate_clients(self):
        has_clients = hasattr(self.ui, 'client_manager') and bool(self.ui.client_manager.clients)
        if not has_clients:
            logger.warning("Client validation failed: no clients available")
        return has_clients

    def _create_action_button(self, text, symbol, bg_color, hover_color):
        btn = QPushButton(text, self.ui.OptionQTableView)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg_color};
                border: 1px solid {'#2E8B57' if 'EE90' in bg_color else '#CD5C5C'};
                border-radius: 3px;
                padding: 5px;
                min-width: 50px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
        """)
        btn.clicked.connect(lambda: self.place_order(symbol, text[0].upper()))
        return btn

    def place_order(self, symbol, action):
        """Place order with IST timestamp logging"""
        order_time = datetime.now(self.IST).strftime('%H:%M:%S IST')
        try:
            logger.info(f"Attempting to place {action} order for {symbol}")
            
            if not self._validate_clients():
                error_msg = f"No active clients at {order_time}"
                QMessageBox.critical(self.ui, "Error", error_msg)
                logger.error(error_msg)
                return

            token = self.symbol_to_token.get(symbol)
            if not token:
                error_msg = f"Invalid symbol: {symbol}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            client = self.ui.client_manager.clients[0][2]
            client.place_order(
                buy_or_sell=action,
                product_type="M",
                exchange="NFO",
                tradingsymbol=symbol,
                quantity=75,
                discloseqty=0,
                price_type="MKT",
                price=0,
                trigger_price=0,
                retention="DAY",
                remarks="NewOrder"
            )
            msg = f"{action} order placed for {symbol} at {order_time}"
            self.ui.log_message("Order", msg)
            QMessageBox.information(self.ui, "Success", msg)
            logger.info(msg)

        except Exception as e:
            error_msg = f"Failed to place order at {order_time}: {str(e)}"
            self.ui.log_message("OrderError", error_msg)
            QMessageBox.critical(self.ui, "Error", error_msg)
            logger.error(error_msg)