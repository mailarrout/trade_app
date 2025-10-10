# modules/payoff_graph.py
"""
PayoffGraphTab - Short Strangle Payoff Visualization
Simplified version with detailed logging
"""

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")

from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import logging
import re
import pandas as pd
from datetime import datetime
import os
import pytz

logger = logging.getLogger(__name__)


class PayoffGraphTab(QWidget):
    def __init__(self, ui, client_manager, parent=None):
        super().__init__(parent)
        logger.info("Initializing PayoffGraphTab - Simplified Version")
        
        self.ui = ui
        self.client_manager = client_manager
        self.ist = pytz.timezone('Asia/Kolkata')  # Add this line
        
        # Initialize client
        self.client = None
        self.client_name = "Unknown"
        
        # Matplotlib figure & canvas with dark background
        self.fig = Figure(facecolor="#002B36", figsize=(10, 6))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)

        # Embed canvas into the QFrame
        logger.info("Embedding matplotlib canvas into PayOffGraph QFrame")
        layout = QVBoxLayout(self.ui.PayOffGraph)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        # Connect adjustment calculation button
        try:
            self.ui.CalculateAdjustmentPointQPushButton.clicked.connect(self.calculate_adjustment_points)
            logger.info("Connected CalculateAdjustmentPointQPushButton")
        except Exception as e:
            logger.warning(f"Could not connect adjustment calculation button: {e}")

        # Poll every 10 seconds
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_graph)
        # self.timer.start(10_000)
        self.timer.start(300_000)
        logger.info("Started payoff graph update timer (10 seconds)")

        # first draw after short delay
        QTimer.singleShot(2000, self.update_graph)

    def _validate_clients(self):
        """Validate clients"""
        try:
            if not hasattr(self, 'client_manager') or not self.client_manager.clients:
                logger.warning("No clients available")
                return False
                
            if self.client is None and self.client_manager.clients:
                self.client = self.client_manager.clients[0][2]
                self.client_name = self.client_manager.clients[0][0]  # Get client name
                logger.info(f"Client set to: {self.client_name} ({type(self.client).__name__})")
                
            return True
            
        except Exception as e:
            logger.error(f"Client validation error: {e}")
            return False

    def fetch_spot_price(self):
        """Fetch current NIFTY spot price"""
        try:
            if not self._validate_clients():
                return None

            quote = self.client.get_quotes("NSE", "26000")
            lp = quote.get("lp") or quote.get("ltp") or quote.get("lastprice")
            if not lp:
                return None
                
            spot_price = float(lp)
            logger.debug(f"Fetched spot price: {spot_price}")
            return spot_price
            
        except Exception as e:
            logger.error(f"Failed to fetch spot price: {e}")
            return None

    def log_all_positions(self, positions):
        """Log all positions for debugging"""
        logger.info("=== ALL POSITIONS DUMP ===")
        logger.info(f"Total positions: {len(positions)}")
        
        for i, pos in enumerate(positions):
            try:
                symbol = pos.get('tsym', 'N/A')
                netqty = int(float(pos.get('netqty', 0)))
                product = pos.get('s_prdt_ali', 'N/A')
                exchange = pos.get('exch', 'N/A')                
                
            except Exception as e:
                logger.error(f"Error logging position {i}: {e}")
        
        logger.info("=== END POSITIONS DUMP ===")

    def extract_strangle_from_positions(self):
        """
        Extract short strangle positions from portfolio
        Returns: (put_strike, put_premium, call_strike, call_premium, total_units)
        """
        try:
            if not self._validate_clients():
                return None, None, None, None, 0

            positions = self.client.get_positions() or []
            logger.info(f"Processing {len(positions)} positions from {self.client_name}")
            
            # Log all positions for debugging
            self.log_all_positions(positions)
            
            # Find short options (netqty < 0)
            short_options = []
            for pos in positions:
                try:
                    net_qty = int(float(pos.get("netqty", 0)))
                    symbol = pos.get("tsym", "").strip()
                    
                    logger.debug(f"Checking position: {symbol}, netqty: {net_qty}")
                    
                    # Skip non-short positions and non-option products
                    if net_qty >= 0:
                        logger.debug(f"Skipping - not short: {symbol}")
                        continue
                    
                    # Check if it's an option (should contain strike price numbers)
                    if not any(char.isdigit() for char in symbol):
                        logger.debug(f"Skipping - no digits in symbol: {symbol}")
                        continue
                        
                    # Parse option type and strike - handle formats like:
                    # NIFTY14OCT25P24750, NIFTY14OCT25C25150
                    opt_type = None
                    strike = None
                    
                    # Method 1: Look for P/C at the end before numbers
                    if symbol.upper().endswith('P') or 'PE' in symbol.upper():
                        opt_type = 'PE'
                    elif symbol.upper().endswith('C') or 'CE' in symbol.upper():
                        opt_type = 'CE'
                    
                    # Method 2: Look for P/C before the strike numbers
                    if not opt_type:
                        # Look for pattern like ...P24750 or ...C25150
                        strike_match = re.search(r'(\d+)(C|CE|P|PE)$', symbol.upper())
                        if strike_match:
                            strike = float(strike_match.group(1))
                            opt_type = 'CE' if strike_match.group(2) in ['C', 'CE'] else 'PE'
                    
                    # Method 3: Extract from the end - last character before numbers
                    if not opt_type:
                        # Reverse the string and find where letters end and numbers begin
                        reversed_symbol = symbol.upper()[::-1]
                        digit_pos = 0
                        for i, char in enumerate(reversed_symbol):
                            if not char.isdigit():
                                digit_pos = i
                                break
                        
                        if digit_pos > 0:
                            last_char = reversed_symbol[digit_pos] if digit_pos < len(reversed_symbol) else ''
                            if last_char == 'P':
                                opt_type = 'PE'
                            elif last_char == 'C':
                                opt_type = 'CE'
                    
                    if not opt_type:
                        logger.warning(f"Could not determine option type for: {symbol}")
                        continue
                    
                    # Extract strike price - find all numbers at the end
                    strike_match = re.findall(r'\d+', symbol)
                    if strike_match:
                        strike = float(strike_match[-1])  # Take the last number group
                    else:
                        logger.warning(f"Could not extract strike from: {symbol}")
                        continue
                    
                    # Get premium
                    premium = 0.0
                    sell_price = float(pos.get("netupldprc", 0)) or float(pos.get("totsellavgprc", 0))
                    if sell_price > 0:
                        premium = sell_price
                    else:
                        premium = float(pos.get("netavgprc", 0))
                    
                    # Get quantity and lot size
                    quantity = abs(net_qty)
                    lotsize = int(float(pos.get("prcftr", pos.get("lot", 75))))
                    
                    option_data = {
                        'type': opt_type,
                        'strike': strike,
                        'premium': premium,
                        'quantity': quantity,
                        'lotsize': lotsize,
                        'symbol': symbol
                    }
                    
                    short_options.append(option_data)
                    
                except Exception as e:
                    logger.error(f"Error processing position {pos.get('tsym', '')}: {e}")
                    continue

            for opt in short_options: 
                # logger.info(f"  - {opt['type']} {opt['strike']} @ {opt['premium']:.2f}")
                pass

            # Separate CE and PE positions
            ce_positions = [p for p in short_options if p['type'] == 'CE']
            pe_positions = [p for p in short_options if p['type'] == 'PE']
            
            if not ce_positions or not pe_positions:
                logger.warning(f"Need both CE and PE positions for strangle. Found CE: {len(ce_positions)}, PE: {len(pe_positions)}")
                return None, None, None, None, 0

            # Take first of each (simplified - you can extend for multiple lots)
            ce_leg = ce_positions[0]
            pe_leg = pe_positions[0]
            
            # Calculate total units (minimum quantity between legs)
            total_units = min(ce_leg['quantity'], pe_leg['quantity']) * ce_leg['lotsize']
            
            logger.info(f"Strangle identified: PE {pe_leg['strike']} @ {pe_leg['premium']:.2f}, CE {ce_leg['strike']} @ {ce_leg['premium']:.2f}, Units: {total_units}")
            
            return (pe_leg['strike'], pe_leg['premium'], 
                    ce_leg['strike'], ce_leg['premium'], total_units)
            
        except Exception as e:
            logger.error(f"Error extracting strangle: {e}")
            return None, None, None, None, 0

    def calculate_short_strangle_payoff(self, spot_price, put_strike, put_premium, call_strike, call_premium, total_units):
        """
        Calculate short strangle payoff - based on your script
        """
        try:
            # Premium calculations
            total_premium_per_unit = put_premium + call_premium
            total_premium_total = total_premium_per_unit * total_units

            # Breakeven points
            lower_breakeven = put_strike - total_premium_per_unit
            upper_breakeven = call_strike + total_premium_per_unit

            # Price range for plotting - extended range
            min_price = min(put_strike, spot_price) - 1500
            max_price = max(call_strike, spot_price) + 1500
            price_range = np.arange(max(0, min_price), max_price + 1, 1)

            # Payoff calculations (SHORT strangle)
            put_payoff_per_unit = put_premium - np.maximum(put_strike - price_range, 0)
            call_payoff_per_unit = call_premium - np.maximum(price_range - call_strike, 0)
            
            strategy_payoff_per_unit = put_payoff_per_unit + call_payoff_per_unit
            strategy_payoff_total = strategy_payoff_per_unit * total_units

            # Max profit
            max_profit_total = total_premium_total


            return price_range, strategy_payoff_total, lower_breakeven, upper_breakeven, max_profit_total
            
        except Exception as e:
            logger.error(f"Error calculating payoff: {e}")
            return None, None, None, None, None

    def update_graph(self):
        """Update the payoff graph"""
        try:
            # Fetch data
            spot = self.fetch_spot_price()
            if spot is None:
                self._draw_empty_chart("No spot price available")
                return

            # Update running spot
            if spot:
                self.ui.RunningSpot.setText(f"Nifty: {spot:.2f}")

            # Extract strangle positions
            put_strike, put_premium, call_strike, call_premium, total_units = self.extract_strangle_from_positions()
            
            if put_strike is None:
                self._draw_empty_chart("No short strangle positions\n(Need short CE & PE)")
                # Update adjustment distances even if no strangle (uses manual adjustments)
                self.update_adjustment_points_distance()
                return

            # Get strategy spot price for calculations
            strategy_spot = self.get_strategy_spot_price()
            calculation_spot = strategy_spot if strategy_spot is not None else spot

            # Calculate payoff
            result = self.calculate_short_strangle_payoff(spot, put_strike, put_premium, call_strike, call_premium, total_units)
            if result[0] is None:
                self._draw_empty_chart("Error calculating payoff")
                self.update_adjustment_points_distance()
                return

            price_range, payoff_total, lower_be, upper_be, max_profit = result
            
            # Auto-fill the range fields with breakeven points
            self.ui.LowerRangeQLineEdit.setText(f"{lower_be:.2f}")
            self.ui.HigherRangeQLineEdit.setText(f"{upper_be:.2f}")
            # Update SpotPriceQLineEdit with strategy spot
            self.ui.SpotPriceQLineEdit.setText(f"{calculation_spot:.2f}")
            
            # Calculate adjustment points (profit bend points) from strategy spot
            adjustment = (upper_be - calculation_spot) / 3
            ce_adjustment = calculation_spot - adjustment  # Left side bend (CE adjustment)
            pe_adjustment = calculation_spot + adjustment  # Right side bend (PE adjustment)
            
            # Update adjustment labels
            self.ui.CESellingAdjustment.setText(f"{ce_adjustment:.2f}")
            self.ui.PESellingAdjustment.setText(f"{pe_adjustment:.2f}")
            
            # Update adjustment distances
            self.update_adjustment_points_distance()
            
            # Draw chart
            self._draw_payoff_chart(spot, price_range, payoff_total, lower_be, upper_be, max_profit,
                                put_strike, put_premium, call_strike, call_premium, total_units,
                                ce_adjustment, pe_adjustment)
            
            logger.info("Payoff graph updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating graph: {e}")
            self._draw_empty_chart("Error updating graph")

    def _draw_payoff_chart(self, spot, price_range, payoff_total, lower_be, upper_be, max_profit,
                put_strike, put_premium, call_strike, call_premium, total_units,
                ce_adjustment, pe_adjustment):
        """Draw the payoff chart with adjustment points and strategy spot"""
        try:
            self.ax.clear()
            self.fig.patch.set_facecolor("#002B36")
            self.ax.set_facecolor("#002B36")

            # Set Y-axis range to show bottom till -30000
            y_min = -30000
            y_max = max_profit * 1.2  # Some padding above max profit
            
            # Plot payoff curve
            self.ax.plot(price_range, payoff_total, color="cyan", linewidth=2, label='Short Strangle Payoff')

            # Key lines
            self.ax.axhline(0, color='white', linestyle='--')
            
            # Current spot line (thick red line)
            self.ax.axvline(spot, color='red', linestyle='-', linewidth=2, 
                        label=f'Current Spot: {spot:.0f}')
            
            # Strategy spot line (thin pink line)
            if hasattr(self, 'strategy_spot_price') and self.strategy_spot_price is not None:
                self.ax.axvline(self.strategy_spot_price, color="#FCFCFC", linestyle='-', 
                            linewidth=1, alpha=0.8, label=f'Strategy Spot: {self.strategy_spot_price:.0f}')
            
            # Breakeven points - only text, no dots
            self.ax.text(lower_be, 0, f' {lower_be:.0f}', color='lime', va='bottom')
            self.ax.text(upper_be, 0, f' {upper_be:.0f}', color='lime', va='bottom')
            
            # Max profit line
            self.ax.hlines(max_profit, price_range.min(), price_range.max(), 
                        colors='yellow', linestyles=':', label=f'Max Profit: {max_profit:.0f}')

            # Strike lines
            self.ax.axvline(put_strike, color='orange', linestyle=':', alpha=0.6, label=f'PE: {put_strike}')
            self.ax.axvline(call_strike, color='magenta', linestyle=':', alpha=0.6, label=f'CE: {call_strike}')
            
            # Adjustment points (profit bend points)
            self.ax.axvline(ce_adjustment, color='lightblue', linestyle='--', alpha=0.7, label=f'CE Adj: {ce_adjustment:.0f}')
            self.ax.axvline(pe_adjustment, color='lightgreen', linestyle='--', alpha=0.7, label=f'PE Adj: {pe_adjustment:.0f}')

            # Chart styling
            self.ax.set_title('Nifty Payoff', color='white', pad=20)
            self.ax.set_xlabel('Nifty Price', color='white')
            self.ax.set_ylabel('P/L (INR)', color='white')
            self.ax.grid(True, color='gray', alpha=0.3)
            self.ax.tick_params(colors='white')
            
            # Set Y-axis limits
            self.ax.set_ylim(y_min, y_max)
            
            # Set X-axis limits to show spot ±1000
            x_min = spot - 1000
            x_max = spot + 1000
            self.ax.set_xlim(x_min, x_max)
            
            # Set x-axis ticks at reasonable intervals (every 200 points)
            x_ticks = np.arange(x_min, x_max + 1, 200)
            self.ax.set_xticks(x_ticks)
            
            # Format x-axis labels as actual prices (24056, 24256, 24456, etc.)
            x_tick_labels = [f'{tick:.0f}' for tick in x_ticks]
            self.ax.set_xticklabels(x_tick_labels)
            
            # Legend - increased rows to accommodate both spot prices
            legend = self.ax.legend(facecolor='#002B36', edgecolor='white', 
                                loc='lower center', bbox_to_anchor=(0.5, -0.4),
                                ncol=3, fontsize=10)
            for text in legend.get_texts():
                text.set_color('white')

            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            logger.error(f"Error drawing chart: {e}")
            raise

    def _draw_empty_chart(self, message):
        """Draw empty chart with message"""
        try:
            self.ax.clear()
            self.fig.patch.set_facecolor("#002B36")
            self.ax.set_facecolor("#002B36")
            self.ax.text(0.5, 0.5, message, color="white", ha="center", va="center", 
                        transform=self.ax.transAxes, fontsize=11)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            for spine in self.ax.spines.values():
                spine.set_visible(False)
            self.canvas.draw()
        except Exception as e:
            logger.error(f"Error drawing empty chart: {e}")

    def calculate_adjustment_points(self):
        """Manual adjustment calculation"""
        try:
            # Get inputs
            lower_text = self.ui.LowerRangeQLineEdit.text().strip()
            higher_text = self.ui.HigherRangeQLineEdit.text().strip()
            
            if not lower_text or not higher_text:
                logger.error("Please enter both lower and higher range values")
                return

            lower_range = float(lower_text)
            higher_range = float(higher_text)
            
            if higher_range <= lower_range:
                logger.error("Higher range must be greater than lower range")
                return

            # Get strategy spot price for calculations
            strategy_spot = self.get_strategy_spot_price()
            if strategy_spot is None:
                logger.error("No strategy spot price available")
                return

            # Update spot price field with strategy spot
            self.ui.SpotPriceQLineEdit.setText(f"{strategy_spot:.2f}")

            # Calculate adjustment points from strategy spot
            adjustment = (higher_range - strategy_spot) / 3
            ce_adjustment = strategy_spot - adjustment  # Left side bend (CE adjustment)
            pe_adjustment = strategy_spot + adjustment  # Right side bend (PE adjustment)

            # Update adjustment labels
            self.ui.CESellingAdjustment.setText(f"{ce_adjustment:.2f}")
            self.ui.PESellingAdjustment.setText(f"{pe_adjustment:.2f}")

            logger.info(f"Adjustments calculated from strategy spot {strategy_spot:.2f}: CE={ce_adjustment:.2f}, PE={pe_adjustment:.2f}")

            self.update_graph()

        except ValueError as e:
            logger.error(f"Invalid number input: {e}")
        except Exception as e:
            logger.error(f"Adjustment calculation failed: {e}")

    def stop_updates(self):
        """Stop timer"""
        try:
            if hasattr(self, 'timer') and self.timer.isActive():
                self.timer.stop()
                logger.info("Payoff graph timer stopped")
        except Exception as e:
            logger.error(f"Error stopping timer: {e}")

    def update_adjustment_points_distance(self):
        """Calculate and display points to CE/PE adjustment from current spot"""
        try:
            # Get current spot price for distance calculations
            spot = self.fetch_spot_price()
            if spot is None:
                logger.debug("No current spot available for distance calculation")
                return

            # Get strategy spot price for chart display (preserve this)
            strategy_spot = self.get_strategy_spot_price()
            if strategy_spot is not None:
                self.strategy_spot_price = strategy_spot  # Keep this for chart drawing

            # Get adjustment values from UI
            ce_adj_text = self.ui.CESellingAdjustment.text().strip()
            pe_adj_text = self.ui.PESellingAdjustment.text().strip()
            
            if not ce_adj_text or not pe_adj_text:
                return
                
            try:
                ce_adj = float(ce_adj_text)
                pe_adj = float(pe_adj_text)
                
                # Calculate distances FROM CURRENT SPOT (your requested change)
                points_to_ce = spot - ce_adj  # Positive if current spot > CE adj
                points_to_pe = pe_adj - spot  # Positive if PE adj > current spot
                
                # Update the label
                self.ui.AdjPointsFromSpot.setText(
                    f"{points_to_ce:.0f} Points to CE Adjustment and {points_to_pe:.0f} Points to PE Adjustment"
                )
                
            except ValueError:
                pass
                
        except Exception as e:
            logger.error(f"Error updating adjustment distances: {e}")

    def get_strategy_spot_price(self):
        """Get spot price from strategy mapping CSV for the latest active position"""
        try:          
            # Get logs directory
            main_app_dir = os.path.dirname(os.path.abspath(__file__))
            logs_dir = os.path.join(main_app_dir, "logs")
            
            # Create filename with current date
            date_str = datetime.now(self.ist).strftime('%Y-%m-%d')
            csv_file = os.path.join(logs_dir, f"{date_str}_strategy_mapping.csv")
            
            if not os.path.exists(csv_file):
                logger.warning(f"Strategy mapping file not found: {csv_file}")
                return None
                
            # Read the CSV
            df = pd.read_csv(csv_file)
            
            # Filter active positions
            active_positions = df[df['status'] == 'Active']
            if active_positions.empty:
                logger.warning("No active positions found in strategy mapping")
                return None
                
            # Convert assigned_date to datetime for proper sorting
            active_positions = active_positions.copy()
            active_positions['assigned_date'] = pd.to_datetime(active_positions['assigned_date'])
            
            # Sort by assigned_date descending (newest first)
            latest_position = active_positions.sort_values('assigned_date', ascending=False).iloc[0]
            strategy_spot = latest_position['spot_price']
            
            return strategy_spot
            
        except Exception as e:
            logger.error(f"Error reading strategy spot price: {e}")
            return None
          