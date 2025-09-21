import os
import time
import math
import logging
import pandas as pd
import pytz
from datetime import datetime, timedelta
from PyQt5.QtChart import QChart, QChartView, QLineSeries, QDateTimeAxis, QValueAxis
from PyQt5.QtCore import Qt, QTimer, QDateTime, QPointF, QMargins
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont
from PyQt5.QtWidgets import QVBoxLayout, QGraphicsSimpleTextItem, QMessageBox, QFileDialog

logger = logging.getLogger(__name__)
symbol_cache = {}


class GraphPlotTab:
    """Handles Sold Price vs LTP Graph"""

    def __init__(self, ui, client_manager):
        logger.info("Initializing GraphPlotTab")
        self.ui = ui
        self.client_manager = client_manager
        self.ist = pytz.timezone('Asia/Kolkata')

        self.series = {'sold': {}, 'ltp': {}, 'sl': {}}
        self.price_labels = {'sold': {}, 'ltp': {}, 'sl': {}}
        self.plot_data = pd.DataFrame(columns=['DateTime', 'Symbol', 'SoldPrice', 'LTP', 'VWAP', 'SLPrice'])
        self._symbol_colors = {}
        self._next_color_index = 0

        self._init_plot()
        self._setup_timer()
        self._wait_for_clients()

        # UI buttons
        self.ui.LoadGraphDataQPushButton.clicked.connect(self.load_graph_data_from_file)
        self.ui.ClearPushButton.clicked.connect(self.clear_graph_and_fields)

        logger.info("GraphPlotTab initialized successfully")

    # ----------------- INITIALIZATION -----------------
    def _init_plot(self):
        """Initialize Sold vs LTP chart"""
        self.chart = QChart()
        self.chart.setTitle("Sold Price vs LTP")
        self.chart.setTitleBrush(QBrush(Qt.white))
        self.chart.setBackgroundBrush(QBrush(QColor("#002B36")))
        self.chart.legend().setAlignment(Qt.AlignRight)
        self.chart.setMargins(QMargins(15, 15, 15, 35))
        self.chart.legend().setLabelBrush(QBrush(Qt.white))

        # X Axis
        self.axisX = QDateTimeAxis()
        self.axisX.setFormat("dd-MM hh:mm:ss")
        self.axisX.setTitleText("Time (IST)")
        self.axisX.setLabelsBrush(QBrush(Qt.white))
        self.axisX.setTitleBrush(QBrush(Qt.white))
        self.axisX.setLinePen(QPen(Qt.white))
        self.axisX.setGridLinePen(QPen(QColor(255, 255, 255, 50)))
        self.axisX.setTickCount(8)
        self.axisX.setLabelsFont(QFont("Arial", 8))
        self.axisX.setLabelsAngle(-45)

        # Y Axis
        self.axisY = QValueAxis()
        self.axisY.setTitleText("Price")
        self.axisY.setLabelsBrush(QBrush(Qt.white))
        self.axisY.setTitleBrush(QBrush(Qt.white))
        self.axisY.setLinePen(QPen(Qt.white))
        self.axisY.setGridLinePen(QPen(QColor(255, 255, 255, 50)))
        self.axisY.setTickCount(10)
        self.axisY.setLabelsFont(QFont("Arial", 8))

        self.chart.addAxis(self.axisX, Qt.AlignBottom)
        self.chart.addAxis(self.axisY, Qt.AlignLeft)

        self.chart_view = QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.Antialiasing)

        layout = QVBoxLayout()
        self.ui.GraphPlot.setLayout(layout)
        layout.addWidget(self.chart_view)

        # Set default ranges
        now = datetime.now(self.ist)
        self.axisX.setRange(
            QDateTime.fromMSecsSinceEpoch(int((now - timedelta(minutes=30)).timestamp() * 1000)),
            QDateTime.fromMSecsSinceEpoch(int((now + timedelta(minutes=30)).timestamp() * 1000))
        )
        self.axisY.setRange(0, 100)

    def _setup_timer(self):
        """Timer for updating graph"""
        self.graph_timer = QTimer()
        self.graph_timer.setInterval(5000)
        self.graph_timer.timeout.connect(self._update_chart)
        self.graph_timer.start()

    def _wait_for_clients(self):
        """Wait until clients are loaded"""
        self.check_client_timer = QTimer()
        self.check_client_timer.setInterval(1000)
        self.check_client_timer.timeout.connect(self._check_client_status)
        self.check_client_timer.start()

    def _check_client_status(self):
        if self._validate_clients():
            self.check_client_timer.stop()
            self._load_initial_data()

    # ----------------- VALIDATION -----------------
    def _validate_clients(self):
        return hasattr(self.ui, 'client_manager') and bool(self.ui.client_manager.clients)

    # ----------------- DATA LOADING -----------------
    def _load_initial_data(self):
        """Load saved CSV if available"""
        try:
            plot_file = self._get_data_filename('sold_positions_data')
            if os.path.exists(plot_file):
                df = pd.read_csv(plot_file, parse_dates=['DateTime'])
                # Convert to IST timezone if not already
                if df['DateTime'].dt.tz is None:
                    df['DateTime'] = df['DateTime'].dt.tz_localize(self.ist)
                else:
                    df['DateTime'] = df['DateTime'].dt.tz_convert(self.ist)
                
                self.plot_data = df
                self._plot_historical_data()
                self._update_axes()
        except Exception as e:
            logger.error(f"Init data load error: {e}")

    def _plot_historical_data(self):
        """Plot all historical data"""
        if self.plot_data.empty:
            return

        for symbol, group in self.plot_data.groupby('Symbol'):
            if symbol not in self.series['sold']:
                initial_price = group['SoldPrice'].iloc[0]
                self._create_series(symbol, initial_price)

            for _, row in group.iterrows():
                ist_dt = self._convert_to_ist_qdatetime(row['DateTime'])
                timestamp = ist_dt.toMSecsSinceEpoch()
                
                self.series['sold'][symbol].append(timestamp, row['SoldPrice'])
                self.series['ltp'][symbol].append(timestamp, row['LTP'])
                self._update_label(symbol, 'sold', row['SoldPrice'])
                self._update_label(symbol, 'ltp', row['LTP'])
                
                if pd.notna(row['SLPrice']):
                    self._update_sl_series(symbol, row['SLPrice'])

    def _convert_to_ist_qdatetime(self, dt):
        """Convert datetime to IST-aware QDateTime"""
        try:
            if isinstance(dt, pd.Timestamp):
                dt = dt.to_pydatetime()
            
            if dt.tzinfo is None:
                dt = self.ist.localize(dt)
            else:
                dt = dt.astimezone(self.ist)
            
            utc_dt = dt.astimezone(pytz.UTC)
            return QDateTime.fromMSecsSinceEpoch(int(utc_dt.timestamp() * 1000))
        except Exception as e:
            logger.error(f"Error converting datetime: {e}")
            return QDateTime.currentDateTime()

    # ----------------- CORE CHART LOGIC -----------------
    def _update_chart(self):
        """Update chart with latest LTPs"""
        try:
            if not self._validate_clients():
                return

            current_positions = self._get_current_positions()
            if current_positions.empty:
                return

            client = self.client_manager.clients[0][2]
            current_time = datetime.now(self.ist)
            new_data = []

            for _, row in current_positions.iterrows():
                token = row.get('Token')
                symbol = row.get('Symbol', '')

                if symbol.endswith('-EQ'):
                    continue

                ltp = self._get_ltp(client, token)
                if ltp is None:
                    continue

                sl_price = self._calculate_sl_price(symbol, row['SellPrice'])
                if sl_price is None:
                    continue

                new_entry = {
                    'DateTime': current_time,
                    'Symbol': symbol,
                    'SoldPrice': row['SellPrice'],
                    'LTP': ltp,
                    'VWAP': self._get_vwap(client, token, symbol),
                    'SLPrice': sl_price
                }
                new_data.append(new_entry)

                if symbol not in self.series['sold']:
                    self._create_series(symbol, row['SellPrice'])

                # Convert current time to IST QDateTime for plotting
                ist_dt = self._convert_to_ist_qdatetime(current_time)
                timestamp = ist_dt.toMSecsSinceEpoch()

                # Update series
                self.series['sold'][symbol].append(timestamp, row['SellPrice'])
                self._update_label(symbol, 'sold', row['SellPrice'])

                self.series['ltp'][symbol].append(timestamp, ltp)
                self._update_label(symbol, 'ltp', ltp)

                # Update SL series
                self._update_sl_series(symbol, sl_price)

            if new_data:
                new_df = pd.DataFrame(new_data)
                self.plot_data = pd.concat([self.plot_data, new_df], ignore_index=True)
                self.plot_data = self.plot_data.drop_duplicates(subset=['Symbol', 'DateTime'], keep='last')
                self._save_data()
                self._update_axes()

        except Exception as e:
            logger.error(f"Chart update error: {e}")

    def _create_series(self, symbol, initial_price):
        """Create new series for a symbol"""
        symbol_color = self._get_symbol_color(symbol)

        # Sold series
        sold_series = QLineSeries()
        sold_series.setName(f"{symbol} Sold")
        sold_pen = QPen(symbol_color, 2)
        sold_pen.setStyle(Qt.SolidLine)
        sold_series.setPen(sold_pen)

        # LTP series
        ltp_series = QLineSeries()
        ltp_series.setName(f"{symbol} LTP")
        ltp_pen = QPen(symbol_color, 2)
        ltp_pen.setStyle(Qt.DotLine)
        ltp_series.setPen(ltp_pen)

        # SL series
        sl_series = QLineSeries()
        sl_series.setName(f"{symbol} SL")
        sl_pen = QPen(symbol_color, 2)
        sl_pen.setStyle(Qt.DashLine)
        sl_series.setPen(sl_pen)

        # Add to chart
        for series in (sold_series, ltp_series, sl_series):
            self.chart.addSeries(series)
            series.attachAxis(self.axisX)
            series.attachAxis(self.axisY)

        # Store references
        self.series['sold'][symbol] = sold_series
        self.series['ltp'][symbol] = ltp_series
        self.series['sl'][symbol] = sl_series

        # Set initial SL price
        sl_price = self._calculate_sl_price(symbol, initial_price)
        if sl_price is not None:
            self._update_sl_series(symbol, sl_price)

    def _get_symbol_color(self, symbol):
        """Return a distinct color for each symbol"""
        if symbol in self._symbol_colors:
            return self._symbol_colors[symbol]

        color_palette = [
            QColor("#1f77b4"), QColor("#ff7f0e"), QColor("#2ca02c"), QColor("#d62728"),
            QColor("#9467bd"), QColor("#8c564b"), QColor("#e377c2"), QColor("#7f7f7f"),
            QColor("#bcbd22"), QColor("#17becf"),
        ]

        if self._next_color_index < len(color_palette):
            color = color_palette[self._next_color_index]
        else:
            hue = int(360 * (self._next_color_index % 20) / 20)
            color = QColor.fromHsv(hue, 200, 220)

        self._symbol_colors[symbol] = color
        self._next_color_index += 1
        return color

    def _update_label(self, symbol, line_type, price):
        """Update price label for a series"""
        if symbol not in self.series[line_type] or self.series[line_type][symbol].count() == 0:
            return

        last_point = self.series[line_type][symbol].at(self.series[line_type][symbol].count() - 1)

        if symbol not in self.price_labels[line_type]:
            label = QGraphicsSimpleTextItem()
            label.setFont(QFont("Arial", 8))
            self.chart.scene().addItem(label)
            self.price_labels[line_type][symbol] = label

        label_item = self.price_labels[line_type][symbol]
        symbol_color = self._get_symbol_color(symbol)
        label_text = f"{symbol} - {line_type.upper()} - {price:.2f}"
        y_offsets = {'sold': -30, 'ltp': -15, 'sl': 0}

        self._place_label(
            label_item,
            label_text,
            symbol_color,
            QPointF(last_point.x(), last_point.y()),
            x_offset=5,
            y_offset=y_offsets.get(line_type, 0)
        )

    def _place_label(self, label_item, text, symbol_color, chart_point, x_offset=5, y_offset=0):
        """Place/update a label on the chart"""
        def do_place():
            pos = self.chart.mapToPosition(chart_point)
            label_item.setText(text)
            label_item.setBrush(QBrush(symbol_color))
            label_item.setPos(pos.x() + x_offset, pos.y() + y_offset)

        QTimer.singleShot(0, do_place)

    def _update_sl_series(self, symbol, sl_price):
        """Update SL series with horizontal line"""
        if symbol not in self.series['sl']:
            return

        # Clear existing points
        self.series['sl'][symbol].clear()

        # Get time range for this symbol
        symbol_data = self.plot_data[self.plot_data['Symbol'] == symbol]
        if not symbol_data.empty:
            min_time = symbol_data['DateTime'].min() - timedelta(minutes=5)
            max_time = symbol_data['DateTime'].max() + timedelta(minutes=5)
        else:
            min_time = datetime.now(self.ist) - timedelta(minutes=30)
            max_time = datetime.now(self.ist) + timedelta(minutes=30)

        # Convert to QDateTime
        min_qdt = self._convert_to_ist_qdatetime(min_time)
        max_qdt = self._convert_to_ist_qdatetime(max_time)

        # Add SL horizontal line
        self.series['sl'][symbol].append(min_qdt.toMSecsSinceEpoch(), sl_price)
        self.series['sl'][symbol].append(max_qdt.toMSecsSinceEpoch(), sl_price)

        # Update label
        if symbol not in self.price_labels['sl']:
            label = QGraphicsSimpleTextItem()
            label.setFont(QFont("Arial", 8, QFont.Bold))
            label.setBrush(QBrush(self._get_symbol_color(symbol)))
            self.chart.scene().addItem(label)
            self.price_labels['sl'][symbol] = label

        label_item = self.price_labels['sl'][symbol]
        label_text = f"{symbol} - SL - {sl_price:.2f}"
        last_point = self.series['sl'][symbol].at(self.series['sl'][symbol].count() - 1)
        
        self._place_label(
            label_item,
            label_text,
            self._get_symbol_color(symbol),
            QPointF(last_point.x(), last_point.y()),
            x_offset=5,
            y_offset=15
        )

    # ----------------- DATA MANAGEMENT -----------------
    def _get_current_positions(self):
        """Get current positions from CSV"""
        try:
            positions_file = self._get_data_filename('positions')
            if not os.path.exists(positions_file):
                return pd.DataFrame()
            df = pd.read_csv(positions_file)
            return df[df['NetQty'] < 0]
        except Exception as e:
            logger.error(f"Error reading positions: {e}")
            return pd.DataFrame()

    def _get_data_filename(self, file_type):
        """Generate standardized filename for data files"""
        main_app_dir = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.join(main_app_dir, "logs")
        date_str = datetime.now(self.ist).strftime('%Y-%m-%d')
        return os.path.join(logs_dir, f"{date_str}_{file_type}.csv")

    def _calculate_sl_price(self, symbol, entry_price):
        """Calculate SL price (20% rule)"""
        try:
            sl_price = math.ceil(entry_price * 1.20 * 20) / 20
            return sl_price
        except Exception as e:
            logger.error(f"SL calc error for {symbol}: {e}")
            return None

    def _get_ltp(self, client, token):
        """Get latest price"""
        try:
            quote = client.get_quotes('NFO', str(token))
            if quote and quote.get('stat') == 'Ok' and 'lp' in quote:
                return float(quote['lp'])
        except Exception as e:
            logger.error(f"LTP fetch error for token {token}: {e}")
        return None

    def _get_vwap(self, client, token, symbol=None):
        """Get VWAP with caching"""
        try:
            now = time.time()
            cache_key = f"{token}_vwap"
            if cache_key in symbol_cache and now - symbol_cache[cache_key]['timestamp'] < 300:
                return symbol_cache[cache_key]['vwap']

            current_time = datetime.now(self.ist)
            market_open = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close = current_time.replace(hour=15, minute=30, second=0, microsecond=0)

            if current_time < market_open:
                market_open = market_open - timedelta(days=1)
                market_close = market_close - timedelta(days=1)

            starttime = int(market_open.astimezone(pytz.UTC).timestamp())
            endtime = int(market_close.astimezone(pytz.UTC).timestamp())

            ret = client.get_time_price_series(
                exchange='NFO', token=str(token),
                starttime=starttime, endtime=endtime, interval=5
            )

            if ret and isinstance(ret, list) and len(ret) > 0:
                vwap = float(ret[-1].get('intvwap', 0))
                symbol_cache[cache_key] = {'vwap': vwap, 'timestamp': now}
                return vwap
        except Exception as e:
            logger.error(f"VWAP fetch error for {symbol}: {e}")
        return None

    def _save_data(self):
        """Save plot data to CSV"""
        try:
            file_path = self._get_data_filename('sold_positions_data')
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            self.plot_data.to_csv(file_path, index=False)
        except Exception as e:
            logger.error(f"Failed to save data: {e}")

    def load_graph_data_from_file(self):
        """Manual load from CSV"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self.ui, "Select Graph Data File", "", "CSV Files (*.csv)"
            )
            if file_path:
                df = pd.read_csv(file_path, parse_dates=['DateTime'])
                # Convert to IST timezone if not already
                if df['DateTime'].dt.tz is None:
                    df['DateTime'] = df['DateTime'].dt.tz_localize(self.ist)
                else:
                    df['DateTime'] = df['DateTime'].dt.tz_convert(self.ist)
                
                self.plot_data = df
                self._clear_plot()
                self._plot_historical_data()
                self._update_axes()
                
                QMessageBox.information(self.ui, "Success", "Graph data loaded successfully")
        except Exception as e:
            logger.error(f"File load error: {e}")
            QMessageBox.critical(self.ui, "Error", f"Could not load file:\n{str(e)}")

    def _clear_plot(self):
        """Clear all series and legend from the chart"""
        self.chart.removeAllSeries()
        self.series = {'sold': {}, 'ltp': {}, 'sl': {}}
        self.price_labels = {'sold': {}, 'ltp': {}, 'sl': {}}
        
        # Remove any remaining items from the scene
        for item in self.chart.scene().items():
            if isinstance(item, QGraphicsSimpleTextItem):
                self.chart.scene().removeItem(item)

    def clear_graph_and_fields(self):
        """Clear chart and reset fields"""
        self._clear_plot()
        self.plot_data = pd.DataFrame(columns=['DateTime', 'Symbol', 'SoldPrice', 'LTP', 'VWAP'])
        # Reset axes to default
        now = datetime.now(self.ist)
        self.axisX.setRange(
            QDateTime.fromMSecsSinceEpoch(int((now - timedelta(minutes=30)).timestamp() * 1000)),
            QDateTime.fromMSecsSinceEpoch(int((now + timedelta(minutes=30)).timestamp() * 1000))
        )
        self.axisY.setRange(0, 100)

    def _update_axes(self):
        """Update chart axes with proper ranges"""
        if self.plot_data.empty:
            return

        # Y-axis range
        min_price = min(self.plot_data[['SoldPrice', 'LTP']].min())
        max_price = max(self.plot_data[['SoldPrice', 'LTP']].max())
        
        if 'SLPrice' in self.plot_data.columns and not self.plot_data['SLPrice'].isna().all():
            sl_min = self.plot_data['SLPrice'].min()
            sl_max = self.plot_data['SLPrice'].max()
            min_price = min(min_price, sl_min)
            max_price = max(max_price, sl_max)

        min_price = max(0, min_price)
        price_range = max_price - min_price
        if price_range == 0:
            price_range = max_price * 0.2 if max_price > 0 else 10
        
        price_padding = max(price_range * 0.15, 5)
        self.axisY.setRange(min_price - price_padding, max_price + price_padding)

        # X-axis range
        min_time = self.plot_data['DateTime'].min()
        max_time = self.plot_data['DateTime'].max()
        time_padding = timedelta(minutes=5)

        min_qdt = self._convert_to_ist_qdatetime(min_time - time_padding)
        max_qdt = self._convert_to_ist_qdatetime(max_time + time_padding)
        
        if min_qdt.isValid() and max_qdt.isValid() and min_qdt < max_qdt:
            self.axisX.setRange(min_qdt, max_qdt)