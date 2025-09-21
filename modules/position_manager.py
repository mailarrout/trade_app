import os
import logging
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtCore import Qt, QTimer
from datetime import datetime, time
from PyQt5.QtWidgets import QTableWidgetItem, QMessageBox, QPushButton, QInputDialog, QMenu
import pandas as pd
from pytz import timezone
IST = timezone('Asia/Kolkata')
import winsound
import sys

logger = logging.getLogger(__name__)

class PositionManager:
    # Configuration constants
    MAX_TOTAL_SELL_QTY = 900
    DEFAULT_TARGET = 30000.0
    DEFAULT_SL = -30000.0
    AUTO_REFRESH_INTERVAL = 10000  # 10 seconds
    EXIT_COOLDOWN = 300000  # 5 minutes
    
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

        # Connect UI signals
        self.ui.PositionRefreshPushButton.clicked.connect(self.update_positions)
        self.ui.SubmitPushButton.clicked.connect(self.update_target_sl)
        self.ui.AllClientsRefreshPushButton.clicked.connect(self.update_all_clients_mtm)
        
        # Context menu for strategy updates
        self.ui.PositionTable.customContextMenuRequested.connect(self._show_context_menu)
        self.ui.PositionTable.setContextMenuPolicy(Qt.CustomContextMenu)
        
        logger.info("PositionManager initialization completed")

    def start_updates(self):
        """Start the auto-refresh timer"""
        logger.info("Starting auto-refresh timer")
        if not self.timer.isActive():
            self.timer.start(self.AUTO_REFRESH_INTERVAL)
            self.auto_refresh()
            self.ui.log_message("System", f"Auto-refresh started ({self.AUTO_REFRESH_INTERVAL//1000}s interval)")

    def stop_updates(self):
        """Stop the auto-refresh timer"""
        logger.info("Stopping auto-refresh timer")
        if self.timer.isActive():
            self.timer.stop()
            self.ui.log_message("System", "Auto-refresh stopped")

    def auto_refresh(self):
        """Automatically refresh positions and check for exit conditions"""
        try:
            update_time = datetime.now(IST)
            self._last_update = update_time.strftime("%H:%M:%S")
            
            self.update_positions()
            self.update_all_clients_mtm()

            current_time = update_time.time()
            if self._enable_market_close_exit:
                if time(15, 15) <= current_time <= time(15, 16) and not self._exit_triggered:
                    logger.info("Market close time reached - triggering exit")
                    self.ui.log_message("System", "Market close time reached - exiting all positions")
                    self.exit_all_positions()
                    self._exit_triggered = True

            self.ui.statusBar().showMessage(f"Last update: {self._last_update}", 5000)

        except Exception as e:
            error_msg = f"Auto-refresh failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.ui.log_message("RefreshError", error_msg)

    def update_positions(self):
        """Main method to update positions table"""
        try:
            logger.info("Starting positions update")
            
            if not self._validate_clients():
                return

            client_name, client_id, primary_client = self.client_manager.clients[0]
            logger.debug(f"Using primary client: {client_name}")

            # Get target and SL values
            target, sl = self._get_target_sl_values()
            if target is None or sl is None:
                return

            # Get positions from client
            positions = self._get_client_positions(primary_client, client_name)
            if positions is None:
                return

            # Update strategy tracking
            self.update_strategy_tracking(positions)

            # Process positions
            rows_data, total_mtm, total_pnl, total_raw_mtm, total_sell_qty = self._process_positions(
                positions, primary_client
            )

            # Safety check: total sell quantity
            if total_sell_qty > self.MAX_TOTAL_SELL_QTY:
                self._handle_trade_limit_violation(total_sell_qty)
                return

            # Update table with processed data
            self._update_positions_table(rows_data)

            # Update MTM display and check exit conditions
            current_mtm = total_mtm + total_pnl
            self.update_mtm_display(current_mtm, total_raw_mtm)
            
            log_msg = f"Positions updated - Valid: {len(rows_data)}, MTM: {total_mtm:.2f}, PnL: {total_pnl:.2f}"
            self.ui.log_message(client_name, log_msg)

            self.save_positions_to_csv(rows_data)
            self.ui.statusBar().showMessage(f"Positions updated for {client_id} ({len(rows_data)} valid)")

            # Check exit conditions
            if not hasattr(self, '_exited_all') or not self._exited_all:
                self._check_exit_conditions(current_mtm, target, sl)

        except Exception as e:
            error_msg = f"Position update failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.ui.log_message("PositionError", error_msg)

    def _validate_clients(self):
        """Validate that clients are available"""
        if not self.client_manager or not self.client_manager.clients:
            logger.warning("No clients available")
            self.ui.log_message("System", "No clients available")
            return False
        return True

    def _get_target_sl_values(self):
        """Get target and SL values from UI"""
        try:
            target_text = self.ui.TargetQEdit.text().strip()
            sl_text = self.ui.SLQLine.text().strip()

            target = float(target_text) if target_text else self.DEFAULT_TARGET
            sl = -abs(float(sl_text)) if sl_text else self.DEFAULT_SL

            # Update UI with defaults if empty
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

    def _get_client_positions(self, client, client_name):
        """Get positions from client with error handling"""
        try:
            positions = client.get_positions()
            if positions is None:
                logger.warning("Client returned None positions")
                self.ui.log_message(client_name, "Failed to get positions")
                return None
            logger.debug(f"Retrieved {len(positions)} positions from client")
            return positions
        except Exception as e:
            error_msg = f"Error fetching positions: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.ui.log_message(client_name, error_msg)
            return None

    def _process_positions(self, positions, client):
        """Process positions and calculate totals"""
        rows_data = []
        total_mtm = 0.0
        total_pnl = 0.0
        total_raw_mtm = 0.0
        total_sell_qty = 0
        invalid_positions = 0

        for pos in positions:
            try:
                symbol = pos.get("tsym", "")
                if not symbol:
                    invalid_positions += 1
                    continue

                # Extract position data
                position_data = self._extract_position_data(pos, client, symbol)
                if not position_data:
                    invalid_positions += 1
                    continue

                total_sell_qty += position_data['sell_qty']
                total_mtm += position_data['mtm']
                total_pnl += position_data['pnl']
                total_raw_mtm += position_data['raw_mtm']

                rows_data.append(position_data)

            except Exception as e:
                invalid_positions += 1
                logger.error(f"Error processing position: {str(e)}", exc_info=True)

        logger.debug(f"Total sell quantity = {total_sell_qty}")
        return rows_data, total_mtm, total_pnl, total_raw_mtm, total_sell_qty

    def _extract_position_data(self, pos, client, symbol):
        """Extract data from a single position"""
        token = pos.get("token", "")
        pe_ce = 'CE' if 'C' in symbol else 'PE' if 'P' in symbol else None

        # Quantity calculations
        buy_qty = self._get_quantity(pos, "buy")
        sell_qty = self._get_quantity(pos, "sell")
        net_qty = int(float(pos.get("netqty", 0)))

        # Price calculations
        buy_price = self._get_price(pos, "buy")
        sell_price = self._get_price(pos, "sell")

        # Market data
        ltp = float(pos.get("lp", 0))
        mtm = float(pos.get("urmtom", 0))
        pnl = float(pos.get("rpnl", 0))
        product = pos.get("s_prdt_ali", "")

        # Raw MTM calculation
        if net_qty < 0:  # Short position
            raw_mtm = (sell_price - ltp) * abs(net_qty)
        elif net_qty > 0:  # Long position  
            raw_mtm = (ltp - buy_price) * net_qty
        else:  # Flat position
            raw_mtm = 0

        # Get strategy
        strategy = self.enhanced_get_strategy_for_position(symbol, token, net_qty)

        return {
            "symbol": symbol, "pe_ce": pe_ce, "token": token,
            "buy_qty": buy_qty, "sell_qty": sell_qty, "net_qty": net_qty,
            "buy_price": buy_price, "sell_price": sell_price, "ltp": ltp,
            "mtm": mtm, "pnl": pnl, "raw_mtm": raw_mtm, "product": product,
            "strategy": strategy, "has_action": net_qty != 0
        }

    def _get_quantity(self, pos, qty_type):
        """Get quantity of specific type from position"""
        qty_map = {
            "buy": ["totbuyqty", "cfbuyqty", "daybuyqty"],
            "sell": ["totsellqty", "cfsellqty", "daysellqty"]
        }
        
        for field in qty_map[qty_type]:
            qty = pos.get(field, 0)
            if qty and float(qty) > 0:
                return int(float(qty))
        return 0

    def _get_price(self, pos, price_type):
        """Get price of specific type from position"""
        price_map = {
            "buy": ["netupldprc", "totbuyavgprc", "cfbuyavgprc", "daybuyavgprc"],
            "sell": ["netupldprc", "totsellavgprc", "cfsellavgprc", "daysellavgprc"]
        }
        
        for field in price_map[price_type]:
            price = pos.get(field, 0)
            if price and float(price) > 0:
                return float(price)
        return 0.0

    def _update_positions_table(self, rows_data):
        """Update the positions table with processed data"""
        self.ui.PositionTable.setRowCount(0)
        self.ui.PositionTable.setColumnCount(13)
        
        # Sort: Shorts → Longs → Flats
        rows_data.sort(key=lambda x: (0 if x["net_qty"] < 0 else 1 if x["net_qty"] > 0 else 2))

        for row_idx, row_data in enumerate(rows_data):
            self.ui.PositionTable.insertRow(row_idx)
            
            # Create table items
            items = self._create_table_items(row_data)
            
            for col, item in enumerate(items):
                self.ui.PositionTable.setItem(row_idx, col, item)

            # Add exit button for active positions
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

    def _handle_trade_limit_violation(self, total_sell_qty):
        """Handle trade limit violation"""
        logger.critical(f"Total sell quantity {total_sell_qty} exceeded limit {self.MAX_TOTAL_SELL_QTY}")
        winsound.Beep(1000, 1000)
        winsound.Beep(1500, 1000)
        self.exit_all_positions()
        QTimer.singleShot(3000, self._close_application)
        self.ui.log_message("TradeLimit", f"Total sell quantity {total_sell_qty} exceeded {self.MAX_TOTAL_SELL_QTY} - exiting")

    def _check_exit_conditions(self, current_mtm, target, sl):
        """Check if exit conditions are met"""
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
        """Reset the exit flag after cooldown"""
        logger.info("Resetting exit flag after cooldown")
        self._exited_all = False
        self.ui.log_message("System", "Exit flag reset, monitoring resumed")

    # ===== MTM Display =====
    def update_mtm_display(self, mtm_value, raw_mtm_value=None):
        """Update the MTMQL and MTMShowQLabel with the combined MTM value"""
        try:
            logger.debug(f"Updating MTM display: {mtm_value}, Raw MTM: {raw_mtm_value}")
            if mtm_value is None:
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
            
            # Add RawMTMQL display
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

    def _set_mtm_displays(self, mtm_text, raw_mtm_value):
        """Set MTM values on all display widgets"""
        self.ui.MTMQL.setText(mtm_text)
        self.ui.MTMShowQLabel.setText(mtm_text)
        self.ui.MTMShowQLabelPayOff.setText(mtm_text)
        
        if raw_mtm_value is not None:
            raw_mtm_text = f"Raw MTM: {raw_mtm_value:+,.2f}"
            self.ui.RawMTMQL.setText(raw_mtm_text)
            self._color_mtm_widget(self.ui.RawMTMQL, raw_mtm_value)
        
        # Color main MTM displays
        if "N/A" not in mtm_text and "Error" not in mtm_text:
            mtm_value = float(mtm_text.split(":")[1].strip())
            self._color_mtm_widget(self.ui.MTMQL, mtm_value)
            self._color_mtm_widget(self.ui.MTMShowQLabel, mtm_value)
            self._color_mtm_widget(self.ui.MTMShowQLabelPayOff, mtm_value)

    def _color_mtm_widget(self, widget, value):
        """Apply coloring to MTM widget based on value"""
        if value > 0:
            widget.setStyleSheet("color: green; font-weight: bold;")
        elif value < 0:
            widget.setStyleSheet("color: red; font-weight: bold;")
        else:
            widget.setStyleSheet("color: white; font-weight: bold;")

    # ===== All Clients MTM =====
    def update_all_clients_mtm(self):
        """Update MTM for all clients"""
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

    def _calculate_client_mtm_pnl(self, positions):
        """Calculate MTM and PnL for a client"""
        try:
            mtm = sum(float(p.get("urmtom", 0)) for p in positions)
            pnl = sum(float(p.get("rpnl", 0)) for p in positions)
            return mtm, pnl
        except (ValueError, AttributeError) as e:
            logger.error(f"Error calculating MTM/PnL: {str(e)}")
            return 0.0, 0.0

    def _create_table_item(self, value, color=None, bold=False):
        """Create a standardized table item"""
        item = QTableWidgetItem(str(value))
        item.setTextAlignment(Qt.AlignCenter)
        
        if color:
            item.setForeground(QColor(color))
        if bold:
            font = QFont()
            font.setBold(True)
            item.setFont(font)
            
        return item

    # ===== Exit Positions =====
    def exit_all_positions(self, client_name=None, symbol=None):
        """Exit positions with optional filters"""
        try:
            if not self._validate_clients():
                return
                
            exit_count = 0
            for name, client_id, client in self.client_manager.clients:
                if client_name and name != client_name:
                    continue
                    
                positions = client.get_positions() or []
                
                for pos in positions:
                    if symbol and pos.get("tsym") != symbol:
                        continue   

                    net_qty = int(float(pos.get("netqty", 0)))
                    if net_qty != 0:
                        exit_count += self._exit_single_position(client, pos, net_qty, name)

            if exit_count == 0:
                logger.info("No positions found to exit with given filters")
                self.ui.log_message("System", "No positions found to exit")
            else:
                logger.info(f"Exit operation completed - {exit_count} positions exited")
                self.ui.log_message("System", f"Exit operation completed - {exit_count} positions exited")
                
            QTimer.singleShot(3000, self.update_positions)

        except Exception as e:
            error_msg = f"Position exit failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.ui.log_message("ExitError", error_msg)

    def _exit_single_position(self, client, pos, net_qty, client_name):
        """Exit a single position"""
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
            error_msg = f"Failed to exit {pos['tsym']}: {str(order_error)}"
            logger.error(error_msg, exc_info=True)
            self.ui.log_message("ExitError", error_msg)
            return 0

    def _get_product_type(self, product_alias):
        """Map product alias to product type"""
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

    # ===== Target/SL Update =====
    def update_target_sl(self):
        """Handle target and SL submission"""
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

    # ===== CSV Export =====
    def save_positions_to_csv(self, positions_data):
        """Save positions to CSV file"""
        try:
            output_data = []
            for pos in positions_data:
                symbol = pos.get("symbol", "")
                token = pos.get("token", "")  
                net_qty = int(pos.get("net_qty", 0))
                
                strategy_value = self.get_strategy_for_position(symbol, token)
                    
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

            # Create filename and path
            current_date = datetime.now(IST).strftime('%Y-%m-%d')
            filename = f"{current_date}_positions.csv"
            main_app_dir = os.path.dirname(os.path.abspath(__file__))
            logs_dir = os.path.join(main_app_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            filepath = os.path.join(logs_dir, filename)

            # Save to CSV
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

    def _close_application(self):
        """Close application due to trade limit violation"""
        logger.critical("Closing application due to trade limit violation")
        self.ui.log_message("System", "Closing application due to trade limit violation")
        if hasattr(self.ui, 'app'):
            self.ui.app.quit()
        else:
            sys.exit(1)

    # ===== Strategy Management =====
    def enhanced_get_strategy_for_position(self, symbol, token, net_qty):
        """Get strategy with user prompting for unassigned positions"""
        if not symbol or not token:
            return ""
        
        strategy = self.get_strategy_for_position(symbol, token)
        
        # Prompt user for unassigned active positions
        if not strategy and net_qty != 0:
            return self.prompt_strategy_selection(symbol, token)
        
        return strategy

    def get_strategy_for_position(self, symbol, token):
        """Get strategy for a specific symbol-token combination"""
        if symbol and token:
            key = self._get_symbol_token_key(symbol, token)
            strategy_data = self._strategy_symbol_token_map.get(key, {})
            
            # Handle both old (string) and new (dict) formats
            if isinstance(strategy_data, dict):
                return strategy_data.get('strategy_name', '')
            else:
                return strategy_data  # For backward compatibility
        return ""

    def _get_symbol_token_key(self, symbol, token):
        """Create unique key for symbol-token combination"""
        return f"{symbol}_{token}"

    def prompt_strategy_selection(self, symbol, token):
        """Prompt user to select strategy for a position"""
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
                key = self._get_symbol_token_key(symbol, token)
                self._strategy_symbol_token_map[key] = {
                    'strategy_name': strategy,
                    'spot_price': self._get_current_spot_price(),
                    'timestamp': datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
                }
                self._save_strategy_mapping()
                
                log_msg = f"User assigned '{strategy}' to {symbol}"
                self.ui.log_message("Strategy", log_msg)
                return strategy
            
            return "Update Required"
            
        except Exception as e:
            error_msg = f"Failed to prompt strategy selection: {str(e)}"
            logger.error(error_msg)
            return "Error"

    def _get_current_spot_price(self):
        """Get current NIFTY spot price"""
        try:
            if hasattr(self.client_manager, 'clients') and self.client_manager.clients:
                client = self.client_manager.clients[0][2]
                quote = client.get_quotes('NSE', '26000')
                
                if not quote or quote.get('stat') != 'Ok':
                    logger.error("Failed to get NIFTY quote")
                    return 0.0
                    
                return float(quote.get('lp', 0))
            return 0.0
                
        except Exception as e:
            logger.error(f"Failed to get spot price: {str(e)}")
            return 0.0

    def update_strategy_tracking(self, positions, force_update=False):
        """Update strategy tracking with current positions"""
        if not self._current_strategy:
            return
            
        current_spot_price = self._get_current_spot_price()
        
        for pos in positions:
            symbol = pos.get("tsym", "")
            token = pos.get("token", "")
            net_qty = int(float(pos.get("netqty", 0)))
            
            if symbol and token and net_qty != 0:
                key = f"{symbol}_{token}"
                self._strategy_symbol_token_map[key] = {
                    'strategy_name': self._current_strategy,
                    'spot_price': current_spot_price,
                    'timestamp': datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
                }
        
        self._save_strategy_mapping()

    def _save_strategy_mapping(self):
        """Save strategy mapping to CSV file"""
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
        """Load strategy mapping from CSV file"""
        mapping = {}
        mapping_file = None
        
        try:
            main_app_dir = os.path.dirname(os.path.abspath(__file__))
            logs_dir = os.path.join(main_app_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            
            csv_files = [f for f in os.listdir(logs_dir) if f.endswith('_strategy_mapping.csv')]
            
            if csv_files:
                csv_files.sort(reverse=True)
                latest_file = csv_files[0]
                mapping_file = os.path.join(logs_dir, latest_file)
                
                df = pd.read_csv(mapping_file)
                has_spot_price = 'spot_price' in df.columns
                
                for _, row in df.iterrows():
                    key = f"{row['symbol']}_{row['token']}"
                    
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
                log_msg = f"Loaded {len(mapping)} strategy mappings"
                self.ui.log_message("Strategy", log_msg)
                
        except FileNotFoundError:
            logger.info("No strategy mapping file found")
        except Exception as e:
            error_msg = f"Failed to load strategy mapping: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.ui.log_message("StrategyError", error_msg)
        
        return mapping

    def get_all_strategy_assignments(self):
        """
        Returns all current strategy assignments with enhanced data.
        Format: [ (strategy_name, symbol, token, net_qty, avg_price, entry_spot_price), ... ]
        """
        assignments = []
        
        if not self._validate_clients():
            return assignments

        try:
            client = self.client_manager.clients[0][2]
            positions = client.get_positions() or []
            
            for pos in positions:
                symbol = pos.get("tsym", "")
                token = pos.get("token", "")
                net_qty = int(float(pos.get("netqty", 0)))
                
                if symbol and token and net_qty != 0:
                    # Get strategy with enhanced data
                    strategy_data = self._strategy_symbol_token_map.get(
                        self._get_symbol_token_key(symbol, token), {}
                    )
                    
                    # Extract strategy name and spot price
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
        """Get average price from position data"""
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

    def _show_context_menu(self, position):
        """Show context menu for strategy updates"""
        try:
            row = self.ui.PositionTable.rowAt(position.y())
            if row < 0:
                return
                
            symbol_item = self.ui.PositionTable.item(row, 0)
            if not symbol_item:
                return
                
            symbol = symbol_item.text()
            
            menu = QMenu()
            strategies = [
                "IBBM Intraday", "General", "Intraday Strangle", 
                "Intraday Straddle", "Strategy920AM", "Monthly Strangle", 
                "Monthly Straddle", "Manual Entry", "Other"
            ]
            
            for strategy in strategies:
                action = menu.addAction(strategy)
                action.triggered.connect(
                    lambda checked, s=strategy, sym=symbol: self._update_strategy_for_symbol(sym, s)
                )
            
            menu.exec_(self.ui.PositionTable.viewport().mapToGlobal(position))
            
        except Exception as e:
            error_msg = f"Context menu error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.ui.log_message("UIError", error_msg)

    def _update_strategy_for_symbol(self, symbol, strategy):
        """Update strategy for a specific symbol"""
        try:
            if not self._validate_clients():
                return
                
            client = self.client_manager.clients[0][2]
            positions = client.get_positions() or []
            
            for pos in positions:
                if pos.get("tsym") == symbol:
                    token = pos.get("token", "")
                    if token:
                        key = self._get_symbol_token_key(symbol, token)
                        self._strategy_symbol_token_map[key] = {
                            'strategy_name': strategy,
                            'spot_price': self._get_current_spot_price(),
                            'timestamp': datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
                        }
                        self._save_strategy_mapping()
                        
                        log_msg = f"Updated {symbol} strategy to: {strategy}"
                        self.ui.log_message("Strategy", log_msg)
                        
                        self.update_positions()
                        return
            
            logger.warning(f"Could not find token for symbol: {symbol}")
            
        except Exception as e:
            error_msg = f"Failed to update strategy for {symbol}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.ui.log_message("StrategyError", error_msg)