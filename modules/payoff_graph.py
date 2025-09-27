import os
import logging
import pandas as pd
import pytz
from datetime import datetime
from PyQt5.QtChart import QChart, QChartView, QLineSeries, QValueAxis
from PyQt5.QtCore import Qt, QTimer, QPointF, QMargins
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QBrush
from PyQt5.QtWidgets import QVBoxLayout, QGraphicsSimpleTextItem, QMessageBox
import winsound
import sip

logger = logging.getLogger(__name__)


class PayoffGraphTab:
    """Handles Payoff Graph and Adjustment Calculations"""

    def __init__(self, ui, client_manager):
        logger.info("Initializing PayoffGraphTab")
        self.ui = ui
        self.client_manager = client_manager
        self.ist = pytz.timezone('Asia/Kolkata')
        self._spot_price_raw = None
        self._spot_price_valid = False
        self._payoff_values = None

        # Connect UI buttons
        self.ui.CalculateAdjustmentPointQPushButton.clicked.connect(self.calculate_adjustment_points)

        # Load historical adjustments first
        self._load_historical_adjustments()

        # Setup timers
        self._setup_spot_timer()
        self._setup_payoff_timer()

        logger.info("PayoffGraphTab initialized successfully")

    def _validate_clients(self):
        """Validate clients with debug logging"""
        has_clients = hasattr(self, 'client_manager') and bool(self.client_manager.clients)
        logger.debug(f"Client validation: {has_clients}")
        if has_clients:
            logger.debug(f"Number of clients: {len(self.client_manager.clients)}")
        return has_clients

    # ----------------- TIMER SETUP -----------------
    def _setup_spot_timer(self):
        """Set up timer for spot price updates"""
        logger.info("Setting up spot price timer")
        self.spot_timer = QTimer()
        self.spot_timer.setInterval(10000)  # 10 seconds
        self.spot_timer.timeout.connect(self._update_spot_price)
        self.spot_timer.start()
        logger.info("Spot price timer started - first update in 10 seconds")
        
        # Trigger first update immediately
        QTimer.singleShot(1000, self._update_spot_price)

    def _setup_payoff_timer(self):
        """Set up timer for payoff chart updates"""
        logger.info("Setting up payoff chart timer")
        self.payoff_timer = QTimer()
        self.payoff_timer.setInterval(10000)  # 10 seconds
        self.payoff_timer.timeout.connect(self._draw_payoff_chart)
        self.payoff_timer.start()
        logger.info("Payoff chart timer started")

    # ----------------- ADJUSTMENT CALCULATIONS -----------------
    def calculate_adjustment_points(self):
        """Calculate CE/PE adjustment ranges with proper validation"""
        try:
            logger.info("Starting adjustment point calculation")
            
            # Validate range inputs
            if not self.ui.LowerRangeQLineEdit.text() or not self.ui.HigherRangeQLineEdit.text():
                error_msg = "Please enter both lower and higher range values"
                logger.error(error_msg)
                QMessageBox.warning(self.ui, "Input Error", error_msg)
                return

            lower_range = float(self.ui.LowerRangeQLineEdit.text())
            higher_range = float(self.ui.HigherRangeQLineEdit.text())
            
            if higher_range <= lower_range:
                error_msg = "Higher range must be greater than lower range"
                logger.error(error_msg)
                QMessageBox.warning(self.ui, "Input Error", error_msg)
                return

            logger.debug(f"Range values: lower={lower_range}, higher={higher_range}")

            # Determine price source
            manual_spot_text = self.ui.SpotPriceQLineEdit.text().strip()
            if manual_spot_text:
                self.calculation_price = float(manual_spot_text)
                source = "Manual Entry"
                logger.debug(f"Using manual spot price: {self.calculation_price}")
            elif self._spot_price_valid:
                self.calculation_price = float(self._spot_price_raw) - 80
                source = "Running Spot (Raw - 80)"
                self.ui.SpotPriceQLineEdit.setText(f"{self.calculation_price:.2f}")
                logger.debug(f"Using running spot price (raw-80): {self.calculation_price}")
            else:
                error_msg = "No valid price source available. Please enter spot price manually."
                logger.error(error_msg)
                QMessageBox.warning(self.ui, "Price Error", error_msg)
                return

            # Calculate adjustments
            adjustment = (higher_range - self.calculation_price) / 3
            ce_adjustment = self.calculation_price - adjustment
            pe_adjustment = self.calculation_price + adjustment

            # Calculate differences
            ce_diff = ce_adjustment - lower_range
            pe_diff = higher_range - pe_adjustment

            # Update UI
            self.ui.CESellingAdjustment.setText(
                f"Till BreakEven: {ce_diff:.2f} | "
                f"CE Adj: {ce_adjustment:.2f} "
                f"(Δ: {adjustment:.2f})"
            )

            self.ui.PESellingAdjustment.setText(
                f"Δ: {adjustment:.2f} "
                f"(PE Adj: {pe_adjustment:.2f}) | "
                f"Till BreakEven: {pe_diff:.2f}"
            )

            # Store values for payoff chart
            self._payoff_values = {
                'lower_range': lower_range,
                'higher_range': higher_range,
                'ce_adj': ce_adjustment,
                'pe_adj': pe_adjustment,
                'current_price': self.calculation_price,
                'adjustment': adjustment
            }

            # Save to CSV
            self._save_adjustment_values(
                spot_price=self.calculation_price,
                lower_range=lower_range,
                higher_range=higher_range,
                ce_adjustment=ce_adjustment,
                pe_adjustment=pe_adjustment,
                adjustment_value=adjustment,
                ce_diff=ce_diff,
                pe_diff=pe_diff,
                source=source
            )

            # Draw payoff chart
            self._draw_payoff_chart()
            logger.info("Adjustment points calculated successfully")

        except ValueError as e:
            error_msg = "Please enter valid numbers for all fields"
            logger.error(f"{error_msg}: {e}")
            QMessageBox.warning(self.ui, "Input Error", error_msg)
        except Exception as e:
            error_msg = f"Failed to calculate adjustments: {str(e)}"
            logger.error(error_msg)
            QMessageBox.critical(self.ui, "Calculation Error", error_msg)

    def _save_adjustment_values(self, spot_price, lower_range, higher_range, ce_adjustment, 
                              pe_adjustment, adjustment_value, ce_diff, pe_diff, source):
        """Save adjustment values to CSV file"""
        try:
            logger.info("Saving adjustment values to CSV")
            
            # Get logs directory
            main_app_dir = os.path.dirname(os.path.abspath(__file__))
            logs_dir = os.path.join(main_app_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            
            # Create filename with current date
            date_str = datetime.now(self.ist).strftime('%Y-%m-%d')
            csv_file = os.path.join(logs_dir, f"{date_str}_adjustment_values.csv")
            
            # Prepare data
            data = {
                'DateTime': [datetime.now(self.ist)],
                'SpotPrice': [spot_price],
                'LowerRange': [lower_range],
                'HigherRange': [higher_range],
                'CEAdjustment': [ce_adjustment],
                'PEAdjustment': [pe_adjustment],
                'AdjustmentValue': [adjustment_value],
                'CEDifference': [ce_diff],
                'PEDifference': [pe_diff],
                'Source': [source]
            }
            
            df = pd.DataFrame(data)
            
            # Append to existing file or create new
            if os.path.exists(csv_file):
                df.to_csv(csv_file, mode='a', header=False, index=False)
            else:
                df.to_csv(csv_file, index=False)
                
            logger.info(f"Adjustment values saved to {csv_file}")
            
        except Exception as e:
            error_msg = f"Failed to save adjustment values: {str(e)}"
            logger.error(error_msg)

    # ----------------- PAYOFF CHART -----------------
    def _draw_payoff_chart(self):
        """Draw the payoff chart with reference lines"""
        try:
            logger.info("Drawing payoff chart")
            
            if not self._payoff_values:
                logger.warning("No payoff values available for chart - trying to load historical data")
                if not self._load_historical_adjustments():
                    logger.warning("No historical adjustment data found either")
                    return False

            vals = self._payoff_values
            lower_range = vals['lower_range']
            higher_range = vals['higher_range']
            ce_adj = vals['ce_adj']
            pe_adj = vals['pe_adj']
            current_price = vals['current_price']
            
            # Check for proximity alerts
            self._check_adjustment_proximity(ce_adj, pe_adj)

            # Create chart
            chart = QChart()
            chart.setTitle("Payoff Chart")
            
            # ✅ DARK THEME STYLING
            chart.setTitleBrush(QBrush(Qt.white))
            chart.setBackgroundBrush(QBrush(QColor("#002B36")))  # Dark background
            chart.legend().setAlignment(Qt.AlignRight)
            chart.setMargins(QMargins(15, 15, 15, 35))
            chart.legend().setLabelBrush(QBrush(Qt.white))

            # Create axes
            axisX = QValueAxis()
            axisY = QValueAxis()
            
            # ✅ AXIS STYLING FOR DARK THEME (lighter grid lines)
            axisX.setLabelsBrush(QBrush(Qt.white))
            axisX.setTitleBrush(QBrush(Qt.white))
            axisX.setLinePen(QPen(Qt.white))
            light_grid_pen = QPen(QColor(255, 255, 255, 35))  # white with ~14% opacity
            axisX.setGridLinePen(light_grid_pen)
            axisX.setMinorGridLinePen(light_grid_pen)  # also dim minor grids
            axisX.setLabelFormat("%.0f")
            axisX.setTitleText("Price Levels")

            axisY.setLabelsBrush(QBrush(Qt.white))
            axisY.setTitleBrush(QBrush(Qt.white))
            axisY.setLinePen(QPen(Qt.white))
            axisY.setGridLinePen(light_grid_pen)
            axisY.setMinorGridLinePen(light_grid_pen)  # also dim minor grids
            axisY.setTitleText("P&L")

            chart.addAxis(axisX, Qt.AlignBottom)
            chart.addAxis(axisY, Qt.AlignLeft)

            # Calculate axis ranges
            min_x = min(lower_range, current_price, ce_adj) - 200
            max_x = max(higher_range, current_price, pe_adj) + 200
            axisX.setRange(min_x, max_x)

            # Calculate premium for Y-axis scaling
            total_premium = self._calculate_total_premium()
            min_y = -total_premium * 1.5 if total_premium > 0 else -10000
            max_y = total_premium * 2.5 if total_premium > 0 else 10000
            axisY.setRange(min_y, max_y)

            # Reference lines (all using hex colors)
            reference_points = [
                ('Calc Price', current_price, "#FFFFFF", Qt.SolidLine, 3),  # White
                ('Live Spot', self._spot_price_raw if self._spot_price_valid else None, 
                "#87CEFA", Qt.DashLine, 3),  # Light blue
                ('Lower Range', lower_range, "#FF6347", Qt.DashLine, 2),  # Tomato red
                ('Higher Range', higher_range, "#FF6347", Qt.DashLine, 2),  # Tomato red
                ('CE Adj', ce_adj, "#FFD700", Qt.DashLine, 2),  # Gold
                ('PE Adj', pe_adj, "#FFD700", Qt.DashLine, 2)   # Gold
            ]

            # Draw reference lines
            for name, value, color_hex, style, width in reference_points:
                if value is None:
                    continue  # Skip if no value
                    
                line = QLineSeries()
                line.setName(f"{name}: {value:.2f}")
                line.append(value, min_y)
                line.append(value, max_y)
                line.setPen(QPen(QColor(color_hex), width, style))
                chart.addSeries(line)
                line.attachAxis(axisX)
                line.attachAxis(axisY)

            # Add zero line
            zero_line = QLineSeries()
            zero_line.setName("Zero Line")
            zero_line.append(min_x, 0)
            zero_line.append(max_x, 0)
            zero_line.setPen(QPen(QColor("#FFFFFF"), 1, Qt.DashLine))  # White
            chart.addSeries(zero_line)
            zero_line.attachAxis(axisX)
            zero_line.attachAxis(axisY)

            # Add premium line if premium exists
            if total_premium > 0:
                premium_line = QLineSeries()
                premium_line.setName(f"Premium: {total_premium:.2f}")
                premium_line.append(min_x, total_premium)
                premium_line.append(max_x, total_premium)
                premium_line.setPen(QPen(QColor("#00FFFF"), 2, Qt.DashLine))  # Cyan
                chart.addSeries(premium_line)
                premium_line.attachAxis(axisX)
                premium_line.attachAxis(axisY)

            # Create chart view
            chart_view = QChartView(chart)
            chart_view.setRenderHint(QPainter.Antialiasing)

            # Update UI layout
            if self.ui.PayOffGraph.layout() is None:
                self.ui.PayOffGraph.setLayout(QVBoxLayout())
            else:
                # Clear existing chart
                while self.ui.PayOffGraph.layout().count():
                    child = self.ui.PayOffGraph.layout().takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()

            self.ui.PayOffGraph.layout().addWidget(chart_view)
            self.payoff_chart_view = chart_view

            # Add labels after chart is rendered
            QTimer.singleShot(100, lambda: self._add_payoff_labels(
                chart_view, current_price, lower_range, higher_range, ce_adj, pe_adj, total_premium))

            logger.info("Payoff chart drawn successfully")
            return True

        except Exception as e:
            error_msg = f"Payoff chart failed: {str(e)}"
            logger.error(error_msg)
            return False

    def _calculate_total_premium(self):
        """Calculate total premium from positions"""
        try:
            total_premium = 0
            positions_file = self._get_data_filename('positions')
            
            if os.path.exists(positions_file):
                df = pd.read_csv(positions_file)
                short_positions = df[df['NetQty'] < 0]
                
                for _, row in short_positions.iterrows():
                    if 'C' in row['Symbol'] or 'P' in row['Symbol']:
                        total_premium += abs(row['NetQty']) * row['SellPrice']
            
            logger.debug(f"Calculated total premium: {total_premium}")
            return total_premium
            
        except Exception as e:
            logger.error(f"Error calculating premium: {str(e)}")
            return 0

    def _add_payoff_labels(self, chart_view, current_price, lower_range, higher_range, ce_adj, pe_adj, total_premium):
        """Add text labels to the payoff chart"""
        try:
            # ✅ Prevent using a deleted chart_view in PyQt5
            if chart_view is None or sip.isdeleted(chart_view):
                return
            if not chart_view.scene():
                return

            # Add labels for key price levels
            self._add_vertical_label(chart_view, current_price, f"Calc: {current_price:.2f}", Qt.blue)

            if self._spot_price_valid:
                self._add_vertical_label(chart_view, self._spot_price_raw, f"Spot: {self._spot_price_raw:.2f}", Qt.green)

            self._add_vertical_label(chart_view, lower_range, f"Low: {lower_range:.2f}", Qt.red)
            self._add_vertical_label(chart_view, higher_range, f"High: {higher_range:.2f}", Qt.red)
            self._add_vertical_label(chart_view, ce_adj, f"CE: {ce_adj:.2f}", QColor(255, 165, 0))
            self._add_vertical_label(chart_view, pe_adj, f"PE: {pe_adj:.2f}", QColor(255, 165, 0))

            if total_premium > 0:
                self._add_vertical_label(chart_view, chart_view.chart().axisX().min() + 50,
                                        f"Premium: {total_premium:.2f}", QColor(0, 102, 102))

            logger.debug("Payoff labels added successfully")

        except Exception as e:
            logger.error(f"Failed to add payoff labels: {str(e)}")


    def _add_vertical_label(self, chart_view, price, text, color):
        """Add a label at the top of a vertical line"""
        try:
            chart = chart_view.chart()
            axisY = chart.axisY()
            if not axisY:
                return
                
            point = chart.mapToPosition(QPointF(price, axisY.max()))
            label = QGraphicsSimpleTextItem(text)
            label.setPos(point.x() - 30, point.y() + 5)
            label.setBrush(QBrush(color))
            label.setFont(QFont("Arial", 8, QFont.Bold))
            chart_view.scene().addItem(label)
            
            logger.debug(f"Added vertical label: {text} at price {price}")
        except Exception as e:
            logger.error(f"Failed to add label at {price}: {str(e)}")

    def _check_adjustment_proximity(self, ce_adj, pe_adj):
        """Check if current price is near adjustment points and alert"""
        try:
            if not self._spot_price_valid:
                return
                
            threshold = 15
            current_price = self._spot_price_raw
            
            if abs(current_price - ce_adj) <= threshold:
                logger.info(f"Current price {current_price:.2f} near CE adjustment {ce_adj:.2f}")
                try:
                    winsound.Beep(1000, 300)  # High beep for CE
                except:
                    logger.warning("Could not play proximity beep sound")
                    
            elif abs(current_price - pe_adj) <= threshold:
                logger.info(f"Current price {current_price:.2f} near PE adjustment {pe_adj:.2f}")
                try:
                    winsound.Beep(800, 300)  # Lower beep for PE
                except:
                    logger.warning("Could not play proximity beep sound")
                    
        except Exception as e:
            logger.error(f"Proximity check error: {e}")

    # ----------------- SPOT PRICE HANDLING -----------------
    def _update_spot_price(self):
        """Update the spot price from NIFTY futures"""
        try:
            logger.debug("Starting spot price update")
            
            if not self._validate_clients():
                error_msg = "No active client available for spot price update"
                logger.warning(error_msg)
                self._spot_price_valid = False
                self.ui.RunningSpot.setText("No client")
                return

            logger.debug("Client validated, getting NIFTY spot price")
            
            client = self.client_manager.clients[0][2]
            logger.debug(f"Using client: {type(client).__name__}")
            
            # Try getting NIFTY spot price (NSE index)
            quote = client.get_quotes('NSE', '26000')
            logger.debug(f"Quote response: {quote}")
            
            if quote and quote.get('stat') == 'Ok' and 'lp' in quote:
                spot_price = float(quote['lp'])
                self._spot_price_raw = spot_price
                self._spot_price_valid = True
                logger.info(f"Spot price updated: {spot_price}")
                self.update_running_spot()
            else:
                error_msg = f"Invalid spot quote response: {quote}"
                logger.warning(error_msg)
                self._spot_price_valid = False
                self.ui.RunningSpot.setText("Invalid quote")
                
        except Exception as e:
            error_msg = f"Spot price update error: {e}"
            logger.error(error_msg)
            self._spot_price_valid = False
            self.ui.RunningSpot.setText("Error")

    def update_running_spot(self):
        """Update the RunningSpot label with current spot price"""
        try:
            if self._spot_price_valid:
                self.ui.RunningSpot.setText(f"Nifty: {self._spot_price_raw:.2f}")
                logger.debug(f"Running spot updated: {self._spot_price_raw}")
            else:
                self.ui.RunningSpot.setText("Trying in next 10 sec")
                logger.debug("Running spot update failed, will retry")
                
        except Exception as e:
            error_msg = f"Running spot update error: {e}"
            logger.error(error_msg)
            self.ui.RunningSpot.setText("Update error")

    # ----------------- UTILITY METHODS -----------------
    def _get_data_filename(self, file_type):
        """Generate standardized filename for data files"""
        main_app_dir = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.join(main_app_dir, "logs")
        date_str = datetime.now(self.ist).strftime('%Y-%m-%d')
        filename = os.path.join(logs_dir, f"{date_str}_{file_type}.csv")
        return filename
    
    def _load_historical_adjustments(self):
        """Load historical adjustment values from the latest CSV file"""
        try:
            logger.info("Loading historical adjustment values")
            main_app_dir = os.path.dirname(os.path.abspath(__file__))
            logs_dir = os.path.join(main_app_dir, "logs")
            
            # Get all adjustment files
            adjustment_files = [f for f in os.listdir(logs_dir) 
                            if f.endswith('_adjustment_values.csv')]
            
            if not adjustment_files:
                logger.warning("No historical adjustment files found")
                return False
                
            # Sort files by date (newest first)
            adjustment_files.sort(reverse=True)
            latest_file = adjustment_files[0]
            csv_file = os.path.join(logs_dir, latest_file)
            
            logger.info(f"Loading from latest adjustment file: {latest_file}")
            
            df = pd.read_csv(csv_file, parse_dates=['DateTime'])
            
            if not df.empty:
                # Get the latest entry
                latest = df.iloc[-1]
                
                # Update UI fields if they're empty
                if not self.ui.SpotPriceQLineEdit.text():
                    self.ui.SpotPriceQLineEdit.setText(f"{latest['SpotPrice']:.2f}")
                
                if not self.ui.LowerRangeQLineEdit.text():
                    self.ui.LowerRangeQLineEdit.setText(f"{latest['LowerRange']:.2f}")
                
                if not self.ui.HigherRangeQLineEdit.text():
                    self.ui.HigherRangeQLineEdit.setText(f"{latest['HigherRange']:.2f}")
                
                # Set payoff values for chart
                self._payoff_values = {
                    'lower_range': latest['LowerRange'],
                    'higher_range': latest['HigherRange'],
                    'ce_adj': latest['CEAdjustment'],
                    'pe_adj': latest['PEAdjustment'],
                    'current_price': latest['SpotPrice'],
                    'adjustment': latest['AdjustmentValue']
                }
                
                logger.info(f"Loaded historical adjustment values from {latest_file}")
                return True
                
        except Exception as e:
            error_msg = f"Failed to load historical adjustments: {str(e)}"
            logger.error(error_msg)
        
        return False