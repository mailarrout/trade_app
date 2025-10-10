# position_workers.py
import logging
from PyQt5.QtCore import QThread, pyqtSignal
from datetime import datetime
from pytz import timezone

IST = timezone('Asia/Kolkata')
logger = logging.getLogger(__name__)

class PositionUpdateWorker(QThread):
    finished = pyqtSignal(list, float, float, float, int)
    error = pyqtSignal(str)
    
    def __init__(self, position_manager, filter_mode="all"):
        super().__init__()
        self.position_manager = position_manager
        self.filter_mode = filter_mode
        
    def run(self):
        try:
            if not self.position_manager._validate_clients():
                self.error.emit("No clients available")
                return
                
            # Get first client safely
            if not self.position_manager.client_manager.clients:
                self.error.emit("No clients available")
                return
                
            client_name, client_id, primary_client = self.position_manager.client_manager.clients[0]
            positions = self.position_manager._fetch_positions(filter_mode=self.filter_mode)
            
            if positions is None:
                self.error.emit("Failed to fetch positions")
                return
                
            rows_data, total_mtm, total_pnl, total_raw_mtm, total_sell_qty = self.position_manager._process_positions(positions, primary_client)
            
            self.finished.emit(rows_data, total_mtm, total_pnl, total_raw_mtm, total_sell_qty)
            
        except Exception as e:
            self.error.emit(f"Position update failed: {str(e)}")

class ClientMTMWorker(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, position_manager):
        super().__init__()
        self.position_manager = position_manager
        
    def run(self):
        try:
            if not self.position_manager._validate_clients():
                self.error.emit("No clients available")
                return
                
            # Clear table in main thread via signal or direct call
            self.position_manager.ui.AllClientsTable.setRowCount(0)
            successful_updates = 0

            for row, (name, client_id, client) in enumerate(self.position_manager.client_manager.clients):
                try:
                    positions = client.get_positions() or []
                    mtm, pnl = self.position_manager._calculate_client_mtm_pnl(positions)

                    # Use signal to update UI in main thread
                    self.position_manager._update_client_row_signal.emit(row, name, mtm, pnl)
                    successful_updates += 1

                except Exception as e:
                    logger.error(f"Error processing client {name}: {str(e)}", exc_info=True)
                    continue

            log_msg = f"Updated {successful_updates}/{len(self.position_manager.client_manager.clients)} client MTMs"
            self.position_manager.ui.log_message("MTM", log_msg)
            self.finished.emit()
            
        except Exception as e:
            self.error.emit(f"Client MTM update failed: {str(e)}")