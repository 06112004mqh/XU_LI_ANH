from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2

class ComponentDetector:
    def __init__(self, model_path):
        # Load model bằng thư viện SAHI
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=model_path,
            confidence_threshold=0.4, # Giảm ngưỡng một chút để bắt nhạy hơn
            device="cuda:0", # Nếu dùng CPU thì sửa thành 'cpu'
        )

    def detect(self, image_path):
        """
        Dùng SAHI để cắt ảnh lớn thành nhiều mảnh nhỏ (slice) và nhận diện
        """
        # Đọc ảnh để lấy kích thước
        img = cv2.imread(image_path)
        height, width = img.shape[:2]

        # Tự động tính toán kích thước slice (mảnh cắt)
        # Với PCB, nên cắt mảnh khoảng 640x640
        slice_height = 640
        slice_width = 640

        # Thực hiện nhận diện dạng Slicing
        result = get_sliced_prediction(
            image_path,
            self.detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=0.2, # Các mảnh chồng lên nhau 20% để không mất linh kiện ở mép
            overlap_width_ratio=0.2
        )

        components = []
        
        # Chuyển đổi kết quả SAHI về format cũ của chúng ta
        for prediction in result.object_prediction_list:
            x1 = int(prediction.bbox.minx)
            y1 = int(prediction.bbox.miny)
            x2 = int(prediction.bbox.maxx)
            y2 = int(prediction.bbox.maxy)
            
            cls_name = prediction.category.name
            
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            components.append({
                'class': cls_name,
                'box': [x1, y1, x2, y2],
                'center': (center_x, center_y)
            })
        
        return components