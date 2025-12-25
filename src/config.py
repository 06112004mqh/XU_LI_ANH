import os

# Roboflow config
API_KEY = "YOUR_ROBOFLOW_API_KEY" # Thay bằng key của bạn
PROJECT_NAME = "pcb-component-detection-lxb5x"
VERSION = 1

# Paths
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best.pt')

# Parameters
CONF_THRESHOLD = 0.5  # Độ tin cậy tối thiểu để nhận diện linh kiện