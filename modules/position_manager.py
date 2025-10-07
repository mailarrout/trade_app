# position_manager.py (refactored)
import os
import logging
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtCore import Qt, QTimer
from datetime import datetime, time
from PyQt5.QtWidgets import QTableWidgetItem, QMessageBox, QPushButton, QInputDialog, QMenu
import pandas as pd
from pytz import timezone

IST = timezone('Asia/Kolkata')
logger = logging.getLogger(__name__)

class PositionManager:
    # --- Constants / configuration ---
    MAX_TOTAL_SELL_QTY = 900
    DEFAULT_TARGET = 30000.0
    DEFAULT_SL = -30000.0
    AUTO_REFRESH_INTERVAL = 10000
    EXIT_COOLDOWN = 300000
    MARKET_OPEN = time(9, 15)
    MARKET_CLOSE = time(15, 30)

    # Strategies and buckets (define once)
    STRATEGIES_INTRADAY = ["IBBM Intraday", "Intraday Straddle", "Intraday Strangle", "Strategy920AM"]
    STRATEGIES_MONTHLY = ["Monthly Strangle", "Monthly Straddle"]
    STRATEGIES_KEEP = STRATEGIES_MONTHLY  # explicit alias for readability


    def __init__(self, ui, client_manager):
        logger.info("Initializing PositionManager")
        self.ui = ui
        self.client_manager = client_manager

        self.timer = QTimer()
        self.timer.timeout.connect(self.auto_refresh)

        self._exit_triggered = False
        self._last_update = None
        self._current_strategy = ""
        self._exited_all = False
        self._enable_market_close_exit = False

        current_date = datetime.now(IST).strftime('%Y-%m-%d')
        self._strategy_mapping_file = f"{current_date}_strategy_mapping.csv"
        self._strategy_symbol_token_map = self._load_strategy_mapping()

        # Connect UI
        self.ui.PositionRefreshPushButton.clicked.connect(self.update_positions)
        self.ui.SubmitPushButton.clicked.connect(self.update_target_sl)
        self.ui.AllClientsRefreshPushButton.clicked.connect(self.update_all_clients_mtm)

        self.ui.PositionTable.customContextMenuRequested.connect(self._show_context_menu)
        self.ui.PositionTable.setContextMenuPolicy(Qt.CustomContextMenu)

        logger.info("PositionManager initialization completed")

    # -------------------------
    # Timer / update lifecycle
    # -------------------------
    def start_updates(self):
        logger.info("Starting auto-refresh timer")
        if not self.timer.isActive():
            if self._is_market_hours():
                self.timer.start(self.AUTO_REFRESH_INTERVAL)
                self.auto_refresh()
                self.ui.log_message("System", f"Auto-refresh started ({self.AUTO_REFRESH_INTERVAL//1000}s interval)")
            else:
                self.ui.log_message("System", "Outside market hours - auto-refresh disabled")
                # Do an initial update outside market hours
                self.update_positions()
                self.update_all_clients_mtm()

    def stop_updates(self):
        logger.info("Stopping auto-refresh timer")
        if self.timer.isActive():
            self.timer.stop()
            self.ui.log_message("System", "Auto-refresh stopped")

    def auto_refresh(self):
        try:
            if not self._is_market_hours():
                current_time = datetime.now(IST).time()
                # Market close exit window (if enabled)
                if self._enable_market_close_exit and time(15, 20) <= current_time <= time(15, 21) and not self._exit_triggered:
                    logger.info("Market close time reached - triggering exit")
                    self.ui.log_message("System", "Market close time reached - exiting all positions")
                    self.exit_all_positions()
                    self._exit_triggered = True
                return

            update_time = datetime.now(IST)
            self._last_update = update_time.strftime("%H:%M:%S")

            self.update_positions()
            self.update_all_clients_mtm()

            current_time = update_time.time()
            if self._enable_market_close_exit and time(15, 20) <= current_time <= time(15, 21) and not self._exit_triggered:
                logger.info("Market close time reached - triggering exit")
                self.ui.log_message("System", "Market close time reached - exiting all positions")
                self.exit_all_positions()
                self._exit_triggered = True

            self.ui.statusBar().showMessage(f"Last update: {self._last_update}", 5000)

        except Exception as e:
            error_msg = f"Auto-refresh failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.ui.log_message("RefreshError", error_msg)

    # -------------------------
    # Top-level operations
    # -------------------------
    def update_positions(self):
        """Main method to refresh positions and update UI/table."""
        try:
            logger.info("Starting positions update")
            if not self._validate_clients():
                return

            client_name, client_id, primary_client = self.client_manager.clients[0]
            logger.debug(f"Using primary client: {client_name}")

            target, sl = self._get_target_sl_values()
            if target is None or sl is None:
                return

            positions = self._fetch_positions(filter_mode="all")  # fetch all positions once
            if positions is None:
                return

            # Run cleanup at top of hour during market hours (optimized)
            if self._is_market_hours():
                current_minute = datetime.now(IST).minute
                if current_minute == 0:
                    cleaned_count = self.cleanup_strategy_mappings_with_positions(positions)
                    if cleaned_count > 0:
                        self.ui.log_message("Cleanup", f"Removed {cleaned_count} inactive strategy mappings")

            rows_data, total_mtm, total_pnl, total_raw_mtm, total_sell_qty = self._process_positions(positions, primary_client)

            if total_sell_qty > self.MAX_TOTAL_SELL_QTY:
                self._handle_trade_limit_violation(total_sell_qty)
                return

            self._update_positions_table(rows_data)

            current_mtm = total_mtm + total_pnl
            self.update_mtm_display(current_mtm, total_raw_mtm)

            log_msg = f"Positions updated - Valid: {len(rows_data)}, MTM: {total_mtm:.2f}, PnL: {total_pnl:.2f}"
            self.ui.log_message(client_name, log_msg)

            self.save_positions_to_csv(rows_data)

            if self._is_market_hours():
                self.ui.statusBar().showMessage(f"Positions updated for {client_id} ({len(rows_data)} valid)", 5000)
            else:
                self.ui.statusBar().showMessage(f"Viewing positions (outside market hours) - {client_id} ({len(rows_data)} valid)", 5000)

            if self._is_market_hours() and (not hasattr(self, '_exited_all') or not self._exited_all):
                self._check_exit_conditions(current_mtm, target, sl)

        except Exception as e:
            error_msg = f"Position update failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.ui.log_message("PositionError", error_msg)

    def update_all_clients_mtm(self):
        """Updates summary MTM table for all configured clients."""
        try:
            if not self._validate_clients():
                return

            self.ui.AllClientsTable.setRowCount(0)
            successful_updates = 0

            for row, (name, client_id, client) in enumerate(self.client_manager.clients):
                try:
                    positions = client.get_positions() or []
                    mtm, pnl = self._calculate_client_mtm_pnl(positions)

                    self.ui.AllClientsTable.insertRow(row)
                    items = [
                        self._create_table_item(name, bold=True),
                        self._create_table_item(f"{mtm:+,.2f}", color='green' if mtm >= 0 else 'red'),
                        self._create_table_item(f"{pnl:+,.2f}", color='green' if pnl >= 0 else 'red')
                    ]

                    for col, item in enumerate(items):
                        self.ui.AllClientsTable.setItem(row, col, item)

                    successful_updates += 1

                except Exception as e:
                    logger.error(f"Error processing client {name}: {str(e)}", exc_info=True)
                    continue

            log_msg = f"Updated {successful_updates}/{len(self.client_manager.clients)} client MTMs"
            self.ui.log_message("MTM", log_msg)

        except Exception as e:
            error_msg = f"Client MTM update failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.ui.log_message("MTMError", error_msg)

    # -------------------------
    # Fetching / processing helpers
    # -------------------------
    def _fetch_positions(self, filter_mode="all"):
        """
        Centralized position fetcher.
        filter_mode: "all" | "active" | "symbols"
        Returns:
            list of position dicts (may be empty)
        """
        try:
            if not self._validate_clients():
                return None

            client = self.client_manager.clients[0][2]
            positions = client.get_positions() or []
            
            if filter_mode == "all":
                return positions
            elif filter_mode == "active":
                return [p for p in positions if int(float(p.get("netqty", 0))) != 0]
            elif filter_mode == "symbols":
                # return flat set of symbols
                return {p.get("tsym", "") for p in positions if p.get("tsym")}
            else:
                return positions

        except Exception as e:
            logger.error(f"Failed to fetch positions: {str(e)}", exc_info=True)
            return None

    def _process_positions(self, positions, client):
        rows_data = []
        total_mtm = 0.0
        total_pnl = 0.0
        total_raw_mtm = 0.0
        total_sell_qty = 0

        for pos in positions:
            try:
                symbol = pos.get("tsym", "")
                if not symbol:
                    continue

                position_data = self._extract_position_data(pos, client, symbol)
                if not position_data:
                    continue

                total_sell_qty += position_data['sell_qty']
                total_mtm += position_data['mtm']
                total_pnl += position_data['pnl']
                total_raw_mtm += position_data['raw_mtm']

                rows_data.append(position_data)

            except Exception as e:
                logger.error(f"Error processing position: {str(e)}", exc_info=True)

        logger.debug(f"Total sell quantity = {total_sell_qty}")
        return rows_data, total_mtm, total_pnl, total_raw_mtm, total_sell_qty

    def _extract_position_data(self, pos, client, symbol):
        token = str(pos.get("token", ""))

        # Determine CE/PE roughly based on symbol characters (original logic kept)
        
        # Better CE/PE detection using dname field
        dname = pos.get("dname", "").strip()
        if dname.endswith('PE'):
            pe_ce = 'PE'
        elif dname.endswith('CE'):
            pe_ce = 'CE'
        else:
            # Fallback to symbol-based detection if dname doesn't have CE/PE
            pe_ce = 'CE' if 'C' in symbol else 'PE' if 'P' in symbol else None

        buy_qty = self._extract_first_valid_value(pos, ["totbuyqty", "cfbuyqty", "daybuyqty"], cast=int)
        sell_qty = self._extract_first_valid_value(pos, ["totsellqty", "cfsellqty", "daysellqty"], cast=int)
        net_qty = int(float(pos.get("netqty", 0)))  # keep same cast rules as before

        buy_price = self._extract_first_valid_value(pos, ["netupldprc", "totbuyavgprc", "cfbuyavgprc", "daybuyavgprc"], cast=float)
        sell_price = self._extract_first_valid_value(pos, ["netupldprc", "totsellavgprc", "cfsellavgprc", "daysellavgprc"], cast=float)

        ltp = float(pos.get("lp", 0) or 0)
        mtm = float(pos.get("urmtom", 0) or 0)
        pnl = float(pos.get("rpnl", 0) or 0)
        product = pos.get("s_prdt_ali", "")

        # Raw MTM approximate: keep same behaviour
        if net_qty < 0:
            raw_mtm = (sell_price - ltp) * abs(net_qty)
        elif net_qty > 0:
            raw_mtm = (ltp - buy_price) * net_qty
        else:
            raw_mtm = 0

        strategy = self.get_strategy_for_position(symbol, token, net_qty)

        return {
            "symbol": symbol,
            "pe_ce": pe_ce,
            "token": token,
            "buy_qty": buy_qty,
            "sell_qty": sell_qty,
            "net_qty": net_qty,
            "buy_price": buy_price,
            "sell_price": sell_price,
            "ltp": ltp,
            "mtm": mtm,
            "pnl": pnl,
            "raw_mtm": raw_mtm,
            "product": product,
            "strategy": strategy,
            "has_action": net_qty != 0
        }

    def _extract_first_valid_value(self, pos, fields, cast=float):
        """
        Unified helper to extract the first valid numeric value from candidate fields.
        Returns cast(value) or a zero-equivalent (0 or 0.0) and handles missing/invalid gracefully.
        """
        for field in fields:
            try:
                v = pos.get(field, 0)
                if v is None:
                    continue
                if isinstance(v, str) and v.strip() == "":
                    continue
                fv = float(v)
                if fv > 0:
                    return int(fv) if cast == int else float(fv)
            except (ValueError, TypeError):
                continue
        # fallback
        return 0 if cast == int else 0.0

    # -------------------------
    # UI Table rendering
    # -------------------------
    def _update_positions_table(self, rows_data):
        # Set columns and clear
        self.ui.PositionTable.setRowCount(0)
        column_count = self.ui.PositionTable.columnCount()

        # Sort sell/long/zero ordering similar to original
        rows_data.sort(key=lambda x: (
        0 if x["net_qty"] < 0 else 1 if x["net_qty"] > 0 else 2,  # net_qty priority
        0 if x.get("pe_ce") == 'CE' else 1 if x.get("pe_ce") == 'PE' else 2  # CE first within each group
        ))

        for row_idx, row_data in enumerate(rows_data):
            self.ui.PositionTable.insertRow(row_idx)

            items = self._create_table_items(row_data)

            for col, item in enumerate(items):
                # If TokenHidden field is stored as a QTableWidgetItem, ensure it's the hidden col
                self.ui.PositionTable.setItem(row_idx, col, item)

            if row_data["has_action"]:
                self._add_exit_button(row_idx, row_data)

    def _create_table_items(self, row_data):
        """Create table items for a row"""
        strategy_value = row_data.get("strategy", "")
        strategy_name = strategy_value.get('strategy_name', 'Update Required') if isinstance(strategy_value, dict) else strategy_value

        strategy_item = QTableWidgetItem(strategy_name)
        strategy_item.setTextAlignment(Qt.AlignCenter)
        
        if strategy_name == "Update Required":
            strategy_item.setForeground(QColor("red"))
            strategy_item.setFont(QFont("", -1, QFont.Bold))

        items = [
            QTableWidgetItem(row_data["symbol"]),
            QTableWidgetItem(row_data["pe_ce"]),
            QTableWidgetItem(str(row_data["buy_qty"])),
            QTableWidgetItem(str(row_data["sell_qty"])),
            QTableWidgetItem(str(row_data["net_qty"])),
            QTableWidgetItem(f"{row_data['sell_price']:.2f}"),
            QTableWidgetItem(f"{row_data['buy_price']:.2f}"),
            QTableWidgetItem(f"{row_data['ltp']:.2f}"),
            QTableWidgetItem(f"{row_data['mtm']:.2f}"),
            QTableWidgetItem(f"{row_data['pnl']:.2f}"),
            QTableWidgetItem(row_data["product"]),
            strategy_item,
        ]

        # Apply coloring
        for col, item in enumerate(items):
            item.setTextAlignment(Qt.AlignCenter)
            if col in [2, 3, 4]:  # Quantity columns
                self._color_quantity_item(item, col, row_data)
            elif col in [8, 9]:  # MTM/PnL columns
                self._color_mtm_item(item, col, row_data)

        return items

    def _color_quantity_item(self, item, col, row_data):
        """Apply coloring to quantity items"""
        value = int(item.text())
        if col == 2 and value > 0:  # Buy quantity
            item.setForeground(QColor("green"))
        elif col == 3 and value > 0:  # Sell quantity
            item.setForeground(QColor("red"))
        elif col == 4:  # Net quantity
            item.setForeground(QColor("green") if value > 0 else QColor("red") if value < 0 else QColor("black"))

    def _color_mtm_item(self, item, col, row_data):
        """Apply coloring to MTM/PnL items"""
        value = float(item.text())
        item.setForeground(QColor("green") if value > 0 else QColor("red") if value < 0 else QColor("black"))

    def _add_exit_button(self, row_idx, row_data):
        """Add exit button for a position row"""
        btn = QPushButton("Exit", self.ui.PositionTable)
        btn.setStyleSheet("""
            QPushButton {
                background-color: #ff6666;
                color: white;
                border: none;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #ff4444;
            }
        """)
        client_name = self.client_manager.clients[0][0] if self.client_manager.clients else ""
        btn.clicked.connect(
            lambda _, n=client_name, s=row_data["symbol"]: self.exit_all_positions(client_name=n, symbol=s)
        )
        self.ui.PositionTable.setCellWidget(row_idx, 12, btn)

    # -------------------------
    # Exit / order logic
    # -------------------------
    def exit_all_positions(self, client_name=None, symbol=None):
        """
        Exit positions for given filters. Keeps monthly strategies and only exits intraday strategies by default.
        If symbol is provided, exit that symbol only.
        """
        try:
            if not self._validate_clients():
                return

            exit_count = 0
            strategies_to_exit = ["IBBM Intraday", "Intraday Straddle"]
            strategies_to_keep = self.STRATEGIES_KEEP

            for name, client_id, client in self.client_manager.clients:
                if client_name and name != client_name:
                    continue

                positions = client.get_positions() or []

                for pos in positions:
                    if symbol and pos.get("tsym") != symbol:
                        continue

                    pos_symbol = pos.get("tsym", "")
                    pos_token = str(pos.get("token", ""))

                    strategy = self.get_strategy_for_position(pos_symbol, pos_token)
                    strategy_name = strategy.get('strategy_name', '') if isinstance(strategy, dict) else str(strategy)

                    # Skip monthly strategies we keep
                    if any(monthly_strat in strategy_name for monthly_strat in strategies_to_keep):
                        logger.info(f"Skipping monthly strategy position: {pos_symbol} - {strategy_name}")
                        continue

                    # Only exit positions with specified intraday strategies
                    if strategy_name not in strategies_to_exit:
                        logger.info(f"Skipping non-target strategy: {pos_symbol} - {strategy_name}")
                        continue

                    net_qty = int(float(pos.get("netqty", 0)))
                    if net_qty != 0:
                        exit_count += self._exit_single_position(client, pos, net_qty, name)

            if exit_count == 0:
                logger.info("No positions found to exit with given filters")
                self.ui.log_message("System", "No positions found to exit")
            else:
                logger.info(f"Exit operation completed - {exit_count} intraday positions exited")
                self.ui.log_message("System", f"Exit operation completed - {exit_count} intraday positions exited")

            QTimer.singleShot(3000, self.update_positions)

        except Exception as e:
            error_msg = f"Position exit failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.ui.log_message("ExitError", error_msg)

    def _exit_single_position(self, client, pos, net_qty, client_name):
        try:
            product_alias = (pos.get("s_prdt_ali") or "").upper()
            product_type = self._get_product_type(product_alias)

            side = "B" if net_qty < 0 else "S"
            symbol = pos["tsym"]

            logger.info(f"Exiting position: {symbol}, Qty: {abs(net_qty)}, Side: {side}, Product: {product_type}")

            client.place_order(
                buy_or_sell=side,
                product_type=product_type,
                exchange="NFO",
                tradingsymbol=symbol,
                quantity=abs(net_qty),
                discloseqty=0,
                price_type="MKT",
                price=0,
                trigger_price=0,
                retention="DAY",
                remarks="ManualExit"
            )

            action = "BUY" if side == "B" else "SELL"
            msg = f"Exited {abs(net_qty)} {symbol} ({action}) for {client_name} (Product: {product_type})"
            self.ui.log_message("Exit", msg)
            return 1

        except Exception as order_error:
            error_msg = f"Failed to exit {pos.get('tsym', '')}: {str(order_error)}"
            logger.error(error_msg, exc_info=True)
            self.ui.log_message("ExitError", error_msg)
            return 0

    def _get_product_type(self, product_alias):
        product_map = {
            "CNC": "C",
            "NRML": "M",
            "MIS": "I",
            "BO": "B",
            "BRACKET ORDER": "B",
            "CO": "H",
            "COVER ORDER": "H"
        }
        return product_map.get(product_alias, "M")

    # -------------------------
    # Target / SL management
    # -------------------------
    def update_target_sl(self):
        target = self.ui.TargetQEdit.text()
        sl = self.ui.SLQLine.text()

        if not target or not sl:
            logger.warning("Target or SL values are empty")
            QMessageBox.warning(self.ui, "Warning", "Please enter both Target and SL values")
            return

        try:
            target_val = float(target)
            sl_val = float(sl)
            self.ui.TargetQLToMTM.setText(target)
            log_msg = f"Target set to {target_val}, SL set to {sl_val}"
            self.ui.log_message("System", log_msg)
            QMessageBox.information(self.ui, "Success", "Target and SL values updated")
        except ValueError as e:
            error_msg = f"Invalid numeric values for Target/SL: {str(e)}"
            logger.error(error_msg, exc_info=True)
            QMessageBox.critical(self.ui, "Error", "Please enter valid numeric values")

    def _get_target_sl_values(self):
        try:
            target_text = self.ui.TargetQEdit.text().strip()
            sl_text = self.ui.SLQLine.text().strip()

            target = float(target_text) if target_text else self.DEFAULT_TARGET
            sl = -abs(float(sl_text)) if sl_text else self.DEFAULT_SL

            # if blanks, populate UI with defaults (keeps original behaviour)
            if not target_text:
                self.ui.TargetQEdit.setText(str(target))
            if not sl_text:
                self.ui.SLQLine.setText(str(abs(sl)))

            return target, sl

        except ValueError as e:
            error_msg = f"Invalid Target/SL values: {str(e)}"
            logger.warning(error_msg)
            self.ui.log_message("System", error_msg)
            return None, None

    # -------------------------
    # MTM display helpers
    # -------------------------
    def update_mtm_display(self, mtm_value, raw_mtm_value=None):
        try:
            logger.debug(f"Updating MTM display: {mtm_value}, Raw MTM: {raw_mtm_value}")
            if mtm_value is None:
                # fallback display
                self.ui.MTMQL.setText("N/A")
                self.ui.MTMShowQLabel.setText("N/A")
                self.ui.MTMShowQLabelPayOff.setText("N/A")
                self.ui.RawMTMQL.setText("N/A")
                logger.warning("MTM value is None, setting displays to N/A")
                return

            mtm_text = f"MTM: {mtm_value:+,.2f}"
            self.ui.MTMQL.setText(mtm_text)
            self.ui.MTMShowQLabel.setText(mtm_text)
            self.ui.MTMShowQLabelPayOff.setText(mtm_text)

            if raw_mtm_value is not None:
                raw_mtm_text = f"Raw MTM: {raw_mtm_value:+,.2f}"
                self.ui.RawMTMQL.setText(raw_mtm_text)
                if raw_mtm_value > 0:
                    self.ui.RawMTMQL.setStyleSheet("color: green; font-weight: bold;")
                elif raw_mtm_value < 0:
                    self.ui.RawMTMQL.setStyleSheet("color: red; font-weight: bold;")
                else:
                    self.ui.RawMTMQL.setStyleSheet("color: white; font-weight: bold;")

            if mtm_value > 0:
                style = "color: green; font-weight: bold;"
            elif mtm_value < 0:
                style = "color: red; font-weight: bold;"
            else:
                style = "color: white; font-weight: bold;"

            self.ui.MTMQL.setStyleSheet(style)
            self.ui.MTMShowQLabel.setStyleSheet(style)
            self.ui.MTMShowQLabelPayOff.setStyleSheet(style)

            logger.debug("MTM display updated successfully")

        except Exception as e:
            error_msg = f"Failed to update MTM display: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.ui.log_message("MTMError", error_msg)
            self.ui.MTMQL.setText("Error")
            self.ui.MTMShowQLabel.setText("Error")
            self.ui.MTMShowQLabelPayOff.setText("Error")
            self.ui.RawMTMQL.setText("Error")

    # -------------------------
    # Client MTM helpers
    # -------------------------
    def _calculate_client_mtm_pnl(self, positions):
        try:
            mtm = sum(float(p.get("urmtom", 0) or 0) for p in positions)
            pnl = sum(float(p.get("rpnl", 0) or 0) for p in positions)
            return mtm, pnl
        except (ValueError, AttributeError) as e:
            logger.error(f"Error calculating MTM/PnL: {str(e)}")
            return 0.0, 0.0

    def _create_table_item(self, value, color=None, bold=False):
        item = QTableWidgetItem(str(value))
        item.setTextAlignment(Qt.AlignCenter)

        if color:
            item.setForeground(QColor(color))
        if bold:
            font = QFont()
            font.setBold(True)
            item.setFont(font)

        return item

    # -------------------------
    # Strategy mapping & persistence
    # -------------------------
    def get_strategy_for_position(self, symbol, token, net_qty=0):
        """
        Return assigned strategy name or 'Position Not Active' / 'Update Required' string.
        If position active and unassigned -> prompt user (original behavior).
        """
        logger.debug(f"get_strategy_for_position called - symbol: {symbol}, token: {token}, net_qty: {net_qty}")

        if not symbol or not token:
            return ""

        # verify active
        if not self._is_position_active(symbol, token):
            logger.debug(f"Position {symbol} not active at broker - skipping strategy assignment")
            return "Position Not Active"

        strategy = self._get_strategy_from_mapping(symbol, token)
        logger.debug(f"Strategy from mapping: {strategy}")

        if not strategy and net_qty != 0:
            logger.debug("No strategy found for active position, prompting user")
            return self.prompt_strategy_selection(symbol, token)

        logger.debug(f"Returning strategy: {strategy}")
        return strategy

    def _get_strategy_from_mapping(self, symbol, token):
        if not symbol or not token:
            logger.debug(f"Symbol or token is empty - symbol: {symbol}, token: {token}")
            return ""

        current_positions = self._get_current_positions_symbols()
        logger.debug(f"Current positions from broker: {current_positions}")
        if symbol not in current_positions:
            logger.debug(f"SYMBOL NOT FOUND - {symbol} not in current positions")
            return ""

        key = self._get_symbol_token_key(symbol, token)
        strategy_data = self._strategy_symbol_token_map.get(key, {})

        if isinstance(strategy_data, dict):
            return strategy_data.get('strategy_name', '')
        else:
            return strategy_data

    def _get_symbol_token_key(self, symbol, token):
        return f"{symbol}_{token}"

    def prompt_strategy_selection(self, symbol, token):
        try:
            strategies = [
                "IBBM Intraday", "General", "Intraday Strangle",
                "Intraday Straddle", "Strategy920AM", "Monthly Strangle",
                "Monthly Straddle", "Manual Entry", "Other"
            ]

            strategy, ok = QInputDialog.getItem(
                self.ui,
                "Select Strategy",
                f"Select strategy for {symbol}:",
                strategies,
                0,
                False
            )

            if ok and strategy:
                # centralize assign logic
                spot_price = self._get_current_spot_price()
                self._assign_strategy(symbol, token, strategy, spot_price)
                log_msg = f"User assigned '{strategy}' to {symbol}"
                self.ui.log_message("Strategy", log_msg)
                return strategy

            return "Update Required"

        except Exception as e:
            error_msg = f"Failed to prompt strategy selection: {str(e)}"
            logger.error(error_msg)
            return "Error"

    def _assign_strategy(self, symbol, token, strategy_name, spot_price=None):
        """
        Centralized function to assign and persist a strategy mapping for a symbol/token.
        """
        try:
            key = self._get_symbol_token_key(symbol, token)
            data = {
                'strategy_name': strategy_name,
                'spot_price': spot_price if spot_price is not None else 0.0,
                'timestamp': datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
            }
            self._strategy_symbol_token_map[key] = data
            self._save_strategy_mapping()
            logger.info(f"Assigned strategy {strategy_name} to {symbol}")
        except Exception as e:
            logger.error(f"Failed to assign strategy for {symbol}: {str(e)}", exc_info=True)

    def _get_current_spot_price(self):
        try:
            if hasattr(self.client_manager, 'clients') and self.client_manager.clients:
                client = self.client_manager.clients[0][2]
                # keep same call as before (user used '26000' param)
                quote = client.get_quotes('NSE', '26000')

                logger.debug(f"Spot price quote response: {quote}")

                if not quote or quote.get('stat') != 'Ok':
                    logger.error(f"Failed to get NIFTY quote: {quote}")
                    return 0.0

                spot_price = float(quote.get('lp', 0) or 0)
                logger.info(f"Current NIFTY spot price: {spot_price}")
                return spot_price

            logger.error("No clients available for spot price")
            return 0.0

        except Exception as e:
            logger.error(f"Failed to get spot price: {str(e)}")
            return 0.0

    def _save_strategy_mapping(self):
        """Persist current _strategy_symbol_token_map to CSV (logs directory)."""
        try:
            main_app_dir = os.path.dirname(os.path.abspath(__file__))
            logs_dir = os.path.join(main_app_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)

            current_date = datetime.now(IST).strftime('%Y-%m-%d')
            self._strategy_mapping_file = f"{current_date}_strategy_mapping.csv"
            mapping_file = os.path.join(logs_dir, self._strategy_mapping_file)

            mapping_data = []
            current_time = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

            for key, strategy_data in self._strategy_symbol_token_map.items():
                parts = key.rsplit('_', 1)
                if len(parts) == 2:
                    symbol, token = parts
                    if isinstance(strategy_data, dict):
                        strategy_name = strategy_data.get('strategy_name', '')
                        spot_price = strategy_data.get('spot_price', 0.0)
                        timestamp = strategy_data.get('timestamp', current_time)
                    else:
                        strategy_name = strategy_data
                        spot_price = 0.0
                        timestamp = current_time

                    mapping_data.append({
                        'symbol': symbol,
                        'token': token,
                        'strategy': strategy_name,
                        'spot_price': spot_price,
                        'assigned_date': timestamp,
                        'status': 'Active'
                    })

            if mapping_data:
                columns = ['symbol', 'token', 'strategy', 'spot_price', 'assigned_date', 'status']
                df = pd.DataFrame(mapping_data, columns=columns)
                df.to_csv(mapping_file, index=False, mode='w')

                log_msg = f"Saved {len(mapping_data)} strategy mappings"
                self.ui.log_message("Strategy", log_msg)

        except Exception as e:
            error_msg = f"Failed to save strategy mapping: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.ui.log_message("StrategyError", error_msg)

    def _load_strategy_mapping(self):
        mapping = {}

        try:
            main_app_dir = os.path.dirname(os.path.abspath(__file__))
            logs_dir = os.path.join(main_app_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)

            csv_files = [f for f in os.listdir(logs_dir) if f.endswith('_strategy_mapping.csv')]

            if not csv_files:
                logger.info("No strategy mapping files found in logs directory")
                return mapping

            csv_files.sort(reverse=True)
            latest_file = csv_files[0]
            mapping_file = os.path.join(logs_dir, latest_file)

            logger.info(f"Loading strategy mappings from: {latest_file}")

            df = pd.read_csv(mapping_file)
            has_spot_price = 'spot_price' in df.columns

            for _, row in df.iterrows():
                symbol = row['symbol']
                token = str(row['token'])
                key = f"{symbol}_{token}"

                if has_spot_price:
                    mapping[key] = {
                        'strategy_name': row['strategy'],
                        'spot_price': float(row.get('spot_price', 0.0)),
                        'timestamp': row.get('assigned_date', '')
                    }
                else:
                    mapping[key] = {
                        'strategy_name': row['strategy'],
                        'spot_price': 0.0,
                        'timestamp': row.get('assigned_date', '')
                    }

            self._strategy_mapping_file = latest_file
            log_msg = f"Loaded {len(mapping)} strategy mappings from {latest_file}"
            logger.info(log_msg)
            # Avoid UI log flood if UI not ready; guard with hasattr
            if hasattr(self, 'ui'):
                self.ui.log_message("Strategy", log_msg)

        except FileNotFoundError:
            logger.info("Strategy mapping file not found")
        except Exception as e:
            error_msg = f"Failed to load strategy mapping: {str(e)}"
            logger.error(error_msg, exc_info=True)
            if hasattr(self, 'ui'):
                self.ui.log_message("StrategyError", error_msg)

        return mapping

    def _get_current_positions_symbols(self):
        """Return set of current position symbols from broker."""
        current_symbols = set()
        try:
            if not self._validate_clients():
                return current_symbols

            client = self.client_manager.clients[0][2]
            positions = client.get_positions() or []

            for pos in positions:
                symbol = pos.get("tsym", "")
                if symbol:
                    current_symbols.add(symbol)

            logger.debug(f"_get_current_positions_symbols returning: {current_symbols}")

        except Exception as e:
            logger.error(f"Error getting current positions: {str(e)}")

        return current_symbols

    # -------------------------
    # Strategy assignment utilities & UI actions
    # -------------------------
    def get_all_strategy_assignments(self):
        assignments = []

        if not self._validate_clients():
            return assignments

        try:
            client = self.client_manager.clients[0][2]
            positions = client.get_positions() or []

            for pos in positions:
                symbol = pos.get("tsym", "")
                token = str(pos.get("token", ""))
                net_qty = int(float(pos.get("netqty", 0)))

                if symbol and token and net_qty != 0:
                    strategy_data = self._strategy_symbol_token_map.get(
                        self._get_symbol_token_key(symbol, token), {}
                    )

                    if isinstance(strategy_data, dict):
                        strategy_name = strategy_data.get('strategy_name', '')
                        spot_price = strategy_data.get('spot_price', 0.0)
                    else:
                        strategy_name = strategy_data
                        spot_price = 0.0

                    avg_price = self._get_position_avg_price(pos)

                    assignments.append((
                        strategy_name, symbol, token, net_qty, avg_price, spot_price
                    ))

        except Exception as e:
            logger.error(f"Error getting strategy assignments: {e}")

        logger.debug(f"Found {len(assignments)} strategy assignments for recovery")
        return assignments

    def _get_position_avg_price(self, pos):
        try:
            return float(pos.get("netupldprc", 0) or
                         pos.get("totbuyavgprc", 0) or
                         pos.get("cfbuyavgprc", 0) or
                         pos.get("daybuyavgprc", 0) or
                         pos.get("totsellavgprc", 0) or
                         pos.get("cfsellavgprc", 0) or
                         pos.get("daysellavgprc", 0) or 0.0)
        except (ValueError, TypeError):
            return 0.0

    # -------------------------
    # Context menu and manual assign
    # -------------------------
    def _show_context_menu(self, position):
        try:
            row = self.ui.PositionTable.rowAt(position.y())
            if row < 0:
                return

            symbol_item = self.ui.PositionTable.item(row, self.IDX_SYMBOL)
            token_item = self.ui.PositionTable.item(row, self.IDX_TOKEN_HIDDEN)
            if not symbol_item or not token_item:
                return

            symbol = symbol_item.text()
            token = token_item.text()

            menu = QMenu()

            assign_action = menu.addAction("Assign Strategy...")
            assign_action.triggered.connect(
                lambda: self.manually_assign_strategy(symbol, token)
            )

            strategies = [
                "IBBM Intraday", "General", "Intraday Strangle",
                "Intraday Straddle", "Strategy920AM", "Monthly Strangle",
                "Monthly Straddle", "Manual Entry", "Other"
            ]

            for strategy in strategies:
                action = menu.addAction(strategy)
                action.triggered.connect(
                    lambda checked, s=strategy, sym=symbol, tok=token: self._update_strategy_for_symbol(sym, tok, s)
                )

            menu.exec_(self.ui.PositionTable.viewport().mapToGlobal(position))

        except Exception as e:
            error_msg = f"Context menu error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.ui.log_message("UIError", error_msg)

    def _update_strategy_for_symbol(self, symbol, token, strategy):
        try:
            key = self._get_symbol_token_key(symbol, token)

            spot_price = self._get_current_spot_price()
            logger.info(f"Capturing spot price {spot_price} for {symbol}")

            self._assign_strategy(symbol, token, strategy, spot_price)

            log_msg = f"Updated {symbol} strategy to: {strategy} (Spot: {spot_price})"
            self.ui.log_message("Strategy", log_msg)

            self.update_positions()

        except Exception as e:
            error_msg = f"Failed to update strategy for {symbol}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.ui.log_message("StrategyError", error_msg)

    def manually_assign_strategy(self, symbol, token):
        try:
            if not symbol or not token:
                return "Invalid symbol/token"

            # Verify active position first
            if not self._is_position_active(symbol, token):
                return "Position not active at broker - cannot assign strategy"

            return self.prompt_strategy_selection(symbol, token)

        except Exception as e:
            error_msg = f"Failed to manually assign strategy: {str(e)}"
            logger.error(error_msg)
            return "Error"

    # -------------------------
    # Position activity & cleanup
    # -------------------------
    def _is_position_active(self, symbol, token):
        """Check if position actually exists and has non-zero quantity at broker"""
        try:
            if not self._validate_clients():
                return False

            client = self.client_manager.clients[0][2]
            positions = client.get_positions() or []

            for pos in positions:
                pos_symbol = pos.get("tsym", "")
                pos_token = str(pos.get("token", ""))
                net_qty = int(float(pos.get("netqty", 0)))

                if pos_symbol == symbol and pos_token == token and net_qty != 0:
                    return True

            return False

        except Exception as e:
            logger.error(f"Error checking position activity: {str(e)}")
            return False

    def cleanup_strategy_mappings(self):
        """Manual cleanup - fetches fresh positions and delegates to optimized cleanup."""
        try:
            if not self._validate_clients():
                return 0

            client = self.client_manager.clients[0][2]
            positions = client.get_positions() or []
            return self.cleanup_strategy_mappings_with_positions(positions)

        except Exception as e:
            error_msg = f"Failed to cleanup strategy mappings: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return 0

    def _get_current_active_positions(self):
        """Get positions that are active (non-zero net quantity)"""
        active_positions = []
        try:
            if not self._validate_clients():
                return active_positions

            client = self.client_manager.clients[0][2]
            positions = client.get_positions() or []

            for pos in positions:
                net_qty = int(float(pos.get("netqty", 0)))
                if net_qty != 0:
                    active_positions.append(pos)

        except Exception as e:
            logger.error(f"Error getting active positions: {str(e)}")

        return active_positions

    def cleanup_strategy_mappings_with_positions(self, positions):
        """Optimized cleanup using already-fetched positions"""
        try:
            active_keys = set()

            for pos in positions:
                symbol = pos.get("tsym", "")
                token = str(pos.get("token", ""))
                net_qty = int(float(pos.get("netqty", 0)))

                if symbol and token and net_qty != 0:
                    active_keys.add(f"{symbol}_{token}")

            # Keep only entries for active keys
            initial_count = len(self._strategy_symbol_token_map)
            self._strategy_symbol_token_map = {
                key: strategy_data
                for key, strategy_data in self._strategy_symbol_token_map.items()
                if key in active_keys
            }
            final_count = len(self._strategy_symbol_token_map)

            removed_count = initial_count - final_count
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} inactive strategy mappings")
                self._save_strategy_mapping()

            return removed_count

        except Exception as e:
            error_msg = f"Failed to cleanup strategy mappings: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return 0

    # -------------------------
    # CSV / persistence utilities
    # -------------------------
    def save_positions_to_csv(self, positions_data):
        """
        Save positions snapshot to CSV (keeps original column order and behavior).
        """
        try:
            output_data = []
            for pos in positions_data:
                symbol = pos.get("symbol", "")
                token = pos.get("token", "")
                net_qty = int(pos.get("net_qty", 0))

                strategy_value = self.get_strategy_for_position(symbol, token)
                # If dict, fetch name
                if isinstance(strategy_value, dict):
                    strategy_value = strategy_value.get('strategy_name', '')

                output_data.append({
                    "Timestamp": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
                    "Symbol": symbol,
                    "Token": str(token),
                    "NetQty": net_qty,
                    "IsNonZero": "Yes" if net_qty != 0 else "No",
                    "BuyQty": int(pos.get("buy_qty", 0)),
                    "SellQty": int(pos.get("sell_qty", 0)),
                    "BuyPrice": float(pos.get("buy_price", 0)),
                    "SellPrice": float(pos.get("sell_price", 0)),
                    "Product": str(pos.get("product", "")),
                    "Strategy": strategy_value,
                    "LTP": float(pos.get("ltp", 0)),
                    "MTM": float(pos.get("mtm", 0)),
                    "PnL": float(pos.get("pnl", 0)),
                    "RawMTM": float(pos.get("raw_mtm", 0))
                })

            current_date = datetime.now(IST).strftime('%Y-%m-%d')
            filename = f"{current_date}_positions.csv"
            main_app_dir = os.path.dirname(os.path.abspath(__file__))
            logs_dir = os.path.join(main_app_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            filepath = os.path.join(logs_dir, filename)

            columns = [
                "Timestamp", "Symbol", "Token", "IsNonZero", "NetQty",
                "BuyQty", "SellQty", "BuyPrice", "SellPrice",
                "Product", "Strategy", "LTP", "MTM", "PnL", "RawMTM"
            ]
            output_df = pd.DataFrame(output_data, columns=columns)
            output_df.to_csv(filepath, mode='w', header=True, index=False, float_format="%.2f")

            log_msg = f"Saved {len(output_data)} positions to {filepath}"
            self.ui.log_message("CSVExport", log_msg)
            return True

        except Exception as e:
            error_msg = f"Failed to save positions to CSV: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.ui.log_message("CSVExportError", error_msg)
            return False

    # -------------------------
    # Misc utilities
    # -------------------------
    def _close_application(self):
        logger.critical("Closing application due to trade limit violation")
        self.ui.log_message("System", "Closing application due to trade limit violation")
        if hasattr(self.ui, 'app'):
            self.ui.app.quit()
        else:
            import sys
            sys.exit(1)

    def _handle_trade_limit_violation(self, total_sell_qty):
        logger.critical(f"Total sell quantity {total_sell_qty} exceeded limit {self.MAX_TOTAL_SELL_QTY}")
        self.exit_all_positions()
        QTimer.singleShot(3000, self._close_application)
        self.ui.log_message("TradeLimit", f"Total sell quantity {total_sell_qty} exceeded {self.MAX_TOTAL_SELL_QTY} - exiting")

    def _check_exit_conditions(self, current_mtm, target, sl):
        if current_mtm >= target:
            logger.info(f"Target reached ({current_mtm:.2f} >= {target:.2f})")
            self.ui.log_message("System", f"Target reached ({current_mtm:.2f} >= {target:.2f})")
            self.exit_all_positions()
            self._exited_all = True
            QTimer.singleShot(self.EXIT_COOLDOWN, self.reset_exit_flag)
        elif current_mtm <= sl:
            logger.info(f"SL triggered ({current_mtm:.2f} <= {sl:.2f})")
            self.ui.log_message("System", f"SL triggered ({current_mtm:.2f} <= {sl:.2f})")
            self.exit_all_positions()
            self._exited_all = True
            QTimer.singleShot(self.EXIT_COOLDOWN, self.reset_exit_flag)

    def reset_exit_flag(self):
        logger.info("Resetting exit flag after cooldown")
        self._exited_all = False
        self.ui.log_message("System", "Exit flag reset, monitoring resumed")

    def _validate_clients(self):
        if not self.client_manager or not getattr(self.client_manager, "clients", None):
            logger.warning("No clients available")
            # guard ui availability
            if hasattr(self, "ui"):
                self.ui.log_message("System", "No clients available")
            return False
        return True

    def _is_market_hours(self):
        try:
            current_time = datetime.now(IST).time()
            current_date = datetime.now(IST)
            is_weekday = current_date.weekday() < 5

            in_market_hours = is_weekday and (self.MARKET_OPEN <= current_time <= self.MARKET_CLOSE)

            status = "Market Hours - Auto-refresh Active" if in_market_hours else "Outside Market Hours - View Only"
            if hasattr(self, '_last_update') and hasattr(self, 'ui'):
                self.ui.statusBar().showMessage(f"{status} | Last update: {self._last_update}", 10000)

            return in_market_hours

        except Exception as e:
            logger.error(f"Error checking market hours: {str(e)}")
            return False
