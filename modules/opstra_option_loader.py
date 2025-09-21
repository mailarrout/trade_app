import os
import time
import pandas as pd
from datetime import datetime
from PyQt5.QtWidgets import QTableView, QAbstractItemView, QFileDialog, QTableWidgetItem
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QColor, QFont
from PyQt5.QtCore import Qt
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import logging

# Get logger for this module
logger = logging.getLogger(__name__)

# -------------------- SETTINGS -------------------- #
USE_WEBSITE_DEFAULT = True  # True = fetch from Opstra, False = load CSV for testing
USERNAME = "amiya000@gmail.com"
PASSWORD = "HcdD+rw7T*P@2xs"

class OpstraOptionLoader:
    def __init__(self, ui, client_manager, use_website=None):
        self.ui = ui
        self.client_manager = client_manager
        self.use_website = use_website if use_website is not None else USE_WEBSITE_DEFAULT
        self.option_chain_df = pd.DataFrame()  # Store loaded option chain
        logger.info("OpstraOptionLoader initialized")

    # -------------------- Selenium Helpers -------------------- #
    def login(self, driver, wait):
        logger.info("Logging into Opstra...")
        driver.get(
            "https://sso.definedge.com/auth/realms/definedge/protocol/openid-connect/auth?"
            "response_type=code&client_id=opstra&redirect_uri=https://opstra.definedge.com/ssologin&login=true"
        )
        wait.until(EC.visibility_of_element_located((By.ID, "username"))).send_keys(USERNAME)
        driver.find_element(By.ID, "password").send_keys(PASSWORD)
        driver.find_element(By.ID, "kc-login").click()
        wait.until(EC.url_contains("opstra.definedge.com"))
        logger.info("Login successful")

    def select_expiry(self, driver, expiry_date):
        try:
            dropdowns = driver.find_elements(By.XPATH, "//div[contains(@class, 'v-select__slot')]")
            for dropdown in dropdowns:
                if dropdown.is_displayed():
                    driver.execute_script("arguments[0].click();", dropdown)
                    time.sleep(2)
                    options = driver.find_elements(By.XPATH, "//div[contains(@class, 'v-list__tile__title')]")
                    for option in options:
                        if expiry_date.upper() in option.text.upper():
                            logger.info(f"Selecting expiry: {option.text}")
                            driver.execute_script("arguments[0].click();", option)
                            time.sleep(3)
                            return True
        except Exception as e:
            logger.error(f"Could not select expiry: {e}")
        return False

    def show_all_strikes(self, driver):
        try:
            all_btns = driver.find_elements(By.XPATH, "//div[@class='v-btn__content' and normalize-space()='All Strikes']")
            for btn in all_btns:
                if btn.is_displayed():
                    logger.info("Found 'All Strikes' button, clicking...")
                    driver.execute_script("arguments[0].click();", btn)
                    time.sleep(6)
                    return True
            logger.warning("'All Strikes' button not found")
        except Exception as e:
            logger.error(f"Error clicking All Strikes: {e}")
        return False

    def extract_option_chain_data(self, driver):
        js_script = """
        const table = document.querySelector('table');
        if (!table) return {headers: [], data: []};
        const rows = table.querySelectorAll('tr');
        const result = {headers: [], data: []};
        const headerCells = rows[0].querySelectorAll('th, td');
        result.headers = Array.from(headerCells).map(cell => cell.textContent.trim());
        for (let i = 1; i < rows.length; i++) {
            const cells = rows[i].querySelectorAll('td');
            const rowData = Array.from(cells).map(cell => cell.textContent.trim());
            if (rowData.length > 0) result.data.push(rowData);
        }
        return result;
        """
        result = driver.execute_script(js_script)
        if result and result["data"]:
            df = pd.DataFrame(result["data"], columns=result["headers"])
            logger.info(f"Extracted {len(df)} rows from option chain")
            return df
        logger.error("No option chain data found")
        return None

    def fetch_option_chain_website(self, expiry_date, max_retries=2):
        attempt = 0
        df = None
        while attempt < max_retries:
            attempt += 1
            logger.info(f"Attempt {attempt} to fetch option chain for {expiry_date}")
            options = webdriver.ChromeOptions()
            options.add_argument("--start-maximized")
            options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
            driver = webdriver.Chrome(options=options)
            wait = WebDriverWait(driver, 60)
            try:
                self.login(driver, wait)
                driver.get("https://opstra.definedge.com/strategy-builder")
                wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                time.sleep(10)

                self.select_expiry(driver, expiry_date)
                self.show_all_strikes(driver)
                df = self.extract_option_chain_data(driver)
                if df is not None:
                    today = datetime.now().strftime("%Y%m%d")
                    os.makedirs("logs", exist_ok=True)
                    filename = os.path.join("logs", f"{today}-option_chain_{expiry_date}_all.csv")
                    df.to_csv(filename, index=False)
                    logger.info(f"Option chain saved: {filename}")
                    break
            except Exception as e:
                logger.error(f"Attempt {attempt} failed: {e}")
            finally:
                driver.quit()
            if df is None and attempt < max_retries:
                logger.info("Retrying in 10 seconds...")
                time.sleep(10)
        return df

    # -------------------- Delta Lookup -------------------- #
    def get_option_delta(self, symbol: str, expiry_date: str) -> float:
        """Get CE or PE delta from option chain DataFrame"""
        try:
            import re
            if self.option_chain_df.empty:
                return 0.0
            # Parse symbol: e.g., NIFTY30SEP25P23850
            match = re.match(r"([A-Z]+)(\d{2}[A-Z]{3}\d{2})([CP])(\d+)", symbol)
            if not match:
                return 0.0
            underlying, sym_expiry, option_type, strike_str = match.groups()
            strike_price = float(strike_str)
            # Match expiry
            if sym_expiry not in expiry_date:
                return 0.0
            # Match strike
            row = self.option_chain_df[self.option_chain_df["StrikePrice"].astype(float) == strike_price]
            if row.empty:
                return 0.0
            # Return delta
            if option_type.upper() == "C":
                return float(row["CallDelta"].values[0])
            elif option_type.upper() == "P":
                return float(row["PutDelta"].values[0])
            return 0.0
        except Exception as e:
            logger.error(f"Error getting delta for {symbol}: {e}")
            return 0.0

    # -------------------- Load and Highlight -------------------- #
    def load_opstra_data(self):
        start_time = time.time()
        try:
            expiry_ui_text = self.ui.ExpiryListDropDown.currentText()
            expiry_date = datetime.strptime(expiry_ui_text, "%d-%b-%Y").strftime("%d%b%Y").upper()

            # -------------------- Load CSV or Website -------------------- #
            if self.use_website:
                df = self.fetch_option_chain_website(expiry_date)
                if df is None:
                    logger.error("Failed to load option chain from website")
                    return
            else:
                # Open file dialog to select CSV
                options = QFileDialog.Options()
                options |= QFileDialog.ReadOnly
                filename, _ = QFileDialog.getOpenFileName(
                    None,
                    "Select Option Chain CSV",
                    os.path.join(os.getcwd(), "logs"),
                    "CSV Files (*.csv);;All Files (*)",
                    options=options
                )
                if not filename:
                    logger.warning("No file selected, aborting.")
                    return
                df = pd.read_csv(filename)

            # -------------------- Save Option Chain for Delta Lookup --------------------
            self.option_chain_df = df  # always assign for Greek table usage

            # -------------------- Fetch Quote -------------------- #
            client = self.client_manager.clients[0][2]
            token = '26000'
            quote_data = client.get_quotes('NSE', str(token))

            rounded_quote = None
            if quote_data and quote_data.get('stat') == 'Ok' and 'lp' in quote_data:
                quote = float(quote_data['lp'])
                rounded_quote = round(quote / 50) * 50
                logger.info(f"Quote: {quote}, Rounded: {rounded_quote}")
            else:
                logger.warning(f"Invalid quote response for token {token}: {quote_data}")

            # -------------------- Populate Option Table -------------------- #
            model = QStandardItemModel()
            model.setColumnCount(len(df.columns))
            model.setHorizontalHeaderLabels(df.columns.tolist())

            for row_idx, row in df.iterrows():
                items = []
                for col_idx, val in enumerate(row):
                    item = QStandardItem(str(val))
                    if rounded_quote is not None and df.columns[col_idx] == "StrikePrice":
                        try:
                            strike_val = float(val)
                            if strike_val == rounded_quote:
                                font = QFont()
                                font.setBold(True)
                                item.setForeground(QColor("red"))
                                item.setFont(font)
                                logger.info(f"Match found at row {row_idx}: StrikePrice {strike_val}")
                        except:
                            pass
                    items.append(item)
                model.appendRow(items)

            self.ui.OpstraOptionQTableView.setModel(model)
            self.ui.OpstraOptionQTableView.resizeColumnsToContents()
            self.ui.OpstraOptionQTableView.setEditTriggers(QAbstractItemView.NoEditTriggers)


            elapsed = time.time() - start_time
            last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if hasattr(self.ui, "OpstraLastUpdatedQLabel"):
                self.ui.OpstraLastUpdatedQLabel.setText(
                    f"Opstra Last Updated: {last_update} | Load Time: {elapsed:.2f}s"
                )
            logger.info(f"Option chain loaded at {last_update} (took {elapsed:.2f} seconds)")

            # -------------------- Update Greek Table -------------------- #
            self._update_positions_greek(expiry_date)  # <-- always call after option chain is loaded

        except Exception as e:
            logger.error(f"Failed to load Opstra data: {e}")

    # -------------------- Greek Table -------------------- #
    def _update_positions_greek(self, expiry_date):
        """Optimized: Update Greek table with position details including delta (scaled by lot size and quantity)."""
        try:
            if not hasattr(self, 'client_manager') or not self.client_manager.clients:
                return

            client_name, client_id, primary_client = self.client_manager.clients[0]

            try:
                positions = primary_client.get_positions()
                if positions is None:
                    return
            except Exception as e:
                self.ui.log_message(client_name, f"Error fetching positions for Greek table: {str(e)}")
                return

            # -------------------- Lot Size Mapping --------------------
            LOT_SIZES = {
                "NIFTY": 75,
                "BANKNIFTY": 15,
                "FINNIFTY": 40,
                "MIDCPNIFTY": 75
            }

            # -------------------- Prepare Delta Lookup --------------------
            delta_lookup = {}
            if hasattr(self, "option_chain_df") and not self.option_chain_df.empty:
                df = self.option_chain_df
                required_columns = ["StrikePrice", "CallDelta", "PutDelta"]
                if all(col in df.columns for col in required_columns):
                    for _, row in df.iterrows():
                        try:
                            strike = float(row["StrikePrice"])
                            call_delta_str = str(row["CallDelta"]).strip()
                            put_delta_str = str(row["PutDelta"]).strip()
                            call_delta = 0.0 if call_delta_str == '-' else float(call_delta_str)
                            put_delta = 0.0 if put_delta_str == '-' else float(put_delta_str)
                            delta_lookup[strike] = {"call": call_delta, "put": put_delta}
                        except Exception:
                            continue

            # -------------------- Reset Table --------------------
            self.ui.GreekTable.setUpdatesEnabled(False)
            self.ui.GreekTable.setRowCount(0)
            self.ui.GreekTable.setColumnCount(6)
            headers = ["Position", "Entry Price", "Current Price", "Exit Price", "P&L", "Delta"]
            self.ui.GreekTable.setHorizontalHeaderLabels(headers)

            rows_data = []
            total_pnl = 0
            total_delta = 0

            import re
            for pos in positions:
                symbol = pos.get("tsym", "")
                if not symbol:
                    continue
                net_qty = int(float(pos.get("netqty") or 0))
                if net_qty == 0:
                    continue

                match = re.match(r"([A-Z]+)(\d{2}[A-Z]{3}\d{2})([CP])(\d+)", symbol)
                if match:
                    underlying, sym_expiry, option_type, strike_str = match.groups()
                    strike_price = float(strike_str)

                    # Short expiry format
                    csv_expiry_short = expiry_date.upper()[:5] + expiry_date.upper()[7:9]

                    if sym_expiry.upper() != csv_expiry_short.upper():
                        delta = 0.0
                    else:
                        if strike_price in delta_lookup:
                            if option_type.upper() == "C":
                                delta = delta_lookup[strike_price]["call"]
                            elif option_type.upper() == "P":
                                delta = delta_lookup[strike_price]["put"]
                            else:
                                delta = 0.0
                            if net_qty < 0:  # SELL position → flip sign
                                delta = -delta
                        else:
                            delta = 0.0
                else:
                    delta = 0.0

                # Lot size scaling
                lot_size = LOT_SIZES.get(underlying.upper(), 1)
                num_lots = abs(net_qty) / lot_size
                delta_contribution = delta * num_lots

                total_delta += delta_contribution

                # P&L calcs
                entry_price = (
                    float(pos.get("totbuyavgprc") or 0) if net_qty > 0 else float(pos.get("totsellavgprc") or 0)
                )
                exit_price = 0.0
                current_price = float(pos.get("lp") or 0)
                mtm = float(pos.get("urmtom") or 0)
                realized_pnl = float(pos.get("rpnl") or 0)
                total_position_pnl = mtm + realized_pnl
                total_pnl += total_position_pnl

                position_desc = f"{'+' if net_qty>0 else '-'}{abs(net_qty)}x {symbol}"
                rows_data.append({
                    "position_desc": position_desc,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "exit_price": exit_price,
                    "pnl": total_position_pnl,
                    "delta": delta_contribution
                })

            # -------------------- Populate Table --------------------
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
                        val = float(item.text())
                        item.setForeground(QColor("green") if val > 0 else QColor("red") if val < 0 else QColor("black"))
                    elif col == 5:
                        val = float(item.text())
                        item.setForeground(QColor("green") if val > 0 else QColor("red") if val < 0 else QColor("black"))
                    self.ui.GreekTable.setItem(row_idx, col, item)

            # -------------------- Add Total Row --------------------
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
                    if col in [4, 5]:
                        item.setFont(QFont("Arial", 10, QFont.Bold))
                        value = total_pnl if col == 4 else total_delta
                        item.setForeground(QColor("green") if value > 0 else QColor("red") if value < 0 else QColor("black"))
                    self.ui.GreekTable.setItem(len(rows_data), col, item)

            self.ui.GreekTable.resizeColumnsToContents()
            self.ui.GreekTable.setUpdatesEnabled(True)

            self.ui.log_message(
                "GreekTable",
                f"Updated {len(rows_data)} positions | Total P&L: {total_pnl:.2f} | Total Delta: {total_delta:.4f}"
            )

        except Exception as e:
            self.ui.log_message("GreekTableError", f"Error updating Greek table: {str(e)}")
            import traceback
            self.ui.log_message("GreekTableError", f"Traceback: {traceback.format_exc()}")