import logging
import os 
from datetime import datetime

# Log file format 
LOG_FILE = f"{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.log"

# Log folder 
log_path = os.path.join(os.getcwd(),"logs",LOG_FILE)

# Creating the directory for the log path
os.makedirs(log_path,exist_ok=True)

LOG_FILE_PATH = os.path.join(log_path,LOG_FILE)

logging.basicConfig(
    level = logging.INFO,
    filename = LOG_FILE_PATH,
    format = "[ %(asctime)s ] %(lineno)s %(name)s - %(levelname)s - %(message)s"
)