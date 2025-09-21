import os
from datetime import datetime
import pandas as pd
import pyotp
from api_helper import ShoonyaApiPy
from PyQt5.QtWidgets import QMessageBox
from pytz import timezone
import logging

# Get logger for this module
logger = logging.getLogger(__name__)

IST = timezone('Asia/Kolkata')

class ClientManager:
    def __init__(self, ui):
        self.ui = ui
        self.clients = []  # Stores (client_name, client_id, client_object)
        self.initialized = False
        logger.info(f"ClientManager initialized at {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}")

    def load_clients(self):
        """Load client credentials from file and initialize API connections"""
        try:
            start_time = datetime.now(IST)
            logger.info("Starting client loading process")
            
            # Check if ClientInfo.txt exists
            if not os.path.exists("ClientInfo.txt"):
                error_msg = "ClientInfo.txt not found"
                logger.error(error_msg)
                QMessageBox.critical(self.ui, "Error", error_msg)
                return False

            logger.debug("ClientInfo.txt file found, reading contents")
            
            # Clear existing clients before loading new ones
            if self.clients:
                logger.info(f"Clearing {len(self.clients)} existing clients")
                for client_name, client_id, client in self.clients:
                    try:
                        client.logout()
                        logger.debug(f"Logged out client: {client_name}")
                    except Exception as e:
                        logger.warning(f"Error logging out client {client_name}: {str(e)}")
                self.clients = []
                logger.info("All existing clients cleared")

            # Read client information from file
            df = pd.read_csv("ClientInfo.txt")
            logger.debug(f"Read ClientInfo.txt with {len(df)} rows")
            
            if df.empty:
                warning_msg = "ClientInfo.txt is empty"
                logger.warning(warning_msg)
                QMessageBox.warning(self.ui, "Warning", warning_msg)
                return False

            logger.info(f"Attempting to login {len(df)} clients")
            success_count = 0
            
            for index, row in df.iterrows():
                client_id = row.get("Client ID", f"Unknown_{index}")
                client_name = row.get("Client Name", client_id)
                
                try:
                    logger.debug(f"Processing client {index+1}/{len(df)}: {client_name}")
                    login_time = datetime.now(IST)
                    
                    # Generate 2FA token
                    token = row.get("token", "")
                    if not token:
                        logger.warning(f"Skipping client {client_name}: No 2FA token provided")
                        continue
                        
                    twoFA = pyotp.TOTP(token).now()
                    logger.debug(f"Generated 2FA token for {client_name}")

                    # Create client instance and attempt login
                    client = ShoonyaApiPy()
                    logger.debug(f"Created API client instance for {client_name}")

                    ret = client.login(
                        userid=row["Client ID"],
                        password=row["Password"],
                        twoFA=twoFA,
                        vendor_code=row["vc"],
                        api_secret=row["app_key"],
                        imei=row["imei"]
                    )

                    if ret and ret.get('stat') == 'Ok':
                        self.clients.append((client_name, row["Client ID"], client))
                        success_count += 1
                        logger.info(f"SUCCESS: {client_name} logged in successfully at {login_time.strftime('%H:%M:%S IST')}")
                    else:
                        error_msg = ret.get('emsg', 'Unknown error') if ret else 'No response from API'
                        logger.error(f"FAILED: {client_name} login failed at {login_time.strftime('%H:%M:%S IST')}: {error_msg}")
                        client = None

                except KeyError as e:
                    logger.error(f"FAILED: Missing required field in ClientInfo.txt for row {index+1}: {str(e)}")
                except Exception as e:
                    logger.error(f"FAILED: {client_name} login failed at {datetime.now(IST).strftime('%H:%M:%S IST')}: {str(e)}")
                    client = None

            # Final status update
            self.initialized = success_count > 0
            duration = (datetime.now(IST) - start_time).total_seconds()
            
            if self.initialized:
                logger.info(f"Successfully loaded {success_count}/{len(df)} clients in {duration:.2f} seconds")
                # Automatically retrieve margin details after successful login
                logger.info("Initiating margin details retrieval after successful login")
                self._day_available_margin_detl()
            else:
                logger.error(f"Failed to load any clients (took {duration:.2f} seconds)")
                
            return self.initialized

        except pd.errors.EmptyDataError:
            error_msg = "ClientInfo.txt is empty or corrupted"
            logger.error(error_msg)
            QMessageBox.critical(self.ui, "Error", error_msg)
            return False
        except Exception as e:
            error_msg = f"Client load failed at {datetime.now(IST).strftime('%H:%M:%S IST')}: {str(e)}"
            logger.critical(error_msg)
            QMessageBox.critical(self.ui, "Error", f"Client load failed: {str(e)}")
            return False

    def load_expiry_dates(self):
        """Load expiry dates from NFO_symbols.txt and populate the dropdown"""
        try:
            start_time = datetime.now(IST)
            logger.info("Starting expiry dates loading process")
            
            if not os.path.exists("NFO_symbols.txt"):
                warning_msg = "NFO_symbols.txt not found"
                logger.warning(warning_msg)
                QMessageBox.warning(self.ui, "Warning", warning_msg)
                return

            logger.debug("NFO_symbols.txt file found, reading contents")
            
            # Configure pandas display options
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', None)

            # Read and filter data
            df = pd.read_csv("NFO_symbols.txt")
            logger.debug(f"Read NFO_symbols.txt with {len(df)} rows")
            
            df = df[(df["Instrument"].str.strip() == "OPTIDX") &
                    (df["Symbol"].str.strip() == "NIFTY")]
            logger.debug(f"Filtered to {len(df)} OPTIDX NIFTY rows")

            # Convert expiry dates to IST timezone-aware datetime
            df['Expiry'] = pd.to_datetime(
                df['Expiry'].astype(str).str.strip(),
                format='%d-%b-%Y',
                errors='coerce'
            ).dt.tz_localize(IST)

            df = df[df['Expiry'].notna()]
            logger.debug(f"After expiry date conversion: {len(df)} valid rows")

            # Filter for relevant dates
            today = datetime.now(IST).date()
            today = pd.to_datetime(today).tz_localize(IST)
            max_day = today + pd.Timedelta(days=50)
            df = df[(df['Expiry'] >= today) & (df['Expiry'] <= max_day)]
            logger.debug(f"After date filtering: {len(df)} rows within 50 days")

            # Get unique expiry dates
            unique_expiries = sorted(
                df['Expiry'].dt.strftime('%d-%b-%Y').unique(),
                key=lambda x: datetime.strptime(x, '%d-%b-%Y')
            )

            # Update UI dropdown
            self.ui.ExpiryListDropDown.clear()
            self.ui.ExpiryListDropDown.addItems(unique_expiries)
            logger.debug(f"Populated dropdown with {len(unique_expiries)} expiry dates")

            duration = (datetime.now(IST) - start_time).total_seconds()
            
            if not unique_expiries:
                warning_msg = f"No valid expiry dates found (took {duration:.2f} seconds)"
                logger.warning(warning_msg)
                self.ui.ExpiryListDropDown.addItem("No valid expiries")
                self.ui.statusBar().showMessage("No expiry dates found")
            else:
                success_msg = f"Loaded {len(unique_expiries)} expiry dates in {duration:.2f} seconds"
                logger.info(success_msg)
                self.ui.statusBar().showMessage(f"Expiry dates loaded at {datetime.now(IST).strftime('%H:%M:%S IST')}")

        except Exception as e:
            error_msg = f"Error loading expiry dates at {datetime.now(IST).strftime('%H:%M:%S IST')}: {str(e)}"
            logger.error(error_msg)
            self.ui.ExpiryListDropDown.addItem("Error loading expiries")
            self.ui.statusBar().showMessage(f"Error loading expiry dates at {datetime.now(IST).strftime('%H:%M:%S IST')}")
            raise

    def _day_available_margin_detl(self):
        """Simple margin details retrieval and save to CSV - appends to same day file"""
        try:
            if not self.clients:
                logger.warning("No clients available for margin details retrieval")
                return False

            logger.info(f"Retrieving margin details for {len(self.clients)} clients")
            os.makedirs("logs", exist_ok=True)
            margin_data = []
            success_count = 0
            
            for client_name, client_id, client in self.clients:
                try:
                    logger.debug(f"Getting margin details for {client_name}")
                    ret = client.get_limits()
                    
                    if ret and ret.get('stat') == 'Ok':
                        data = {
                            'client_name': client_name, 
                            'client_id': client_id,
                            'retrieval_time': datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')
                        }
                        data.update(ret)
                        margin_data.append(data)
                        success_count += 1
                        logger.info(f"SUCCESS: Margin details retrieved for {client_name}")
                    else:
                        error_msg = ret.get('emsg', 'Unknown error') if ret else 'No response from API'
                        logger.error(f"FAILED: Margin retrieval failed for {client_name}: {error_msg}")
                        
                except Exception as e:
                    logger.error(f"FAILED: Exception while getting margin for {client_name}: {str(e)}")

            # Save margin data to file
            if margin_data:
                filename = f"{datetime.now(IST).strftime('%Y-%m-%d')}-day_available_margin_detl.csv"
                filepath = os.path.join("logs", filename)
                
                if os.path.exists(filepath):
                    # Append to existing file
                    pd.DataFrame(margin_data).to_csv(filepath, mode='a', header=False, index=False)
                    logger.info(f"Appended margin details for {success_count} clients to {filename}")
                else:
                    # Create new file
                    pd.DataFrame(margin_data).to_csv(filepath, index=False)
                    logger.info(f"Created new margin details file: {filename} with {success_count} clients")
                    
                return True
            else:
                logger.warning("No margin data was collected from any client")
                return False
                
        except Exception as e:
            logger.error(f"Margin details retrieval failed: {str(e)}")
            return False