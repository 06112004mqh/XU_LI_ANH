from ultralytics import YOLO
import os

def train_model():
    # 1. Đường dẫn tới file yaml
    # Lưu ý: Nếu chạy lỗi, hãy điền đường dẫn tuyệt đối (VD: "D:/PCB_Project/data/data.yaml")
    yaml_path = os.path.join(os.getcwd(), 'data', 'data.yaml')

    # 2. Load model
    # Dùng yolov8m.pt (medium) để nhận diện linh kiện nhỏ tốt hơn bản nano
    model = YOLO('yolov8m.pt')  

    # 3. Train
    print("Bắt đầu train...")
    results = model.train(
        data=yaml_path,
        epochs=50,       # Dataset này tốt nên 50 epochs là đủ demo
        imgsz=1280,
        batch=2,        # Nếu tràn RAM GPU thì giảm xuống 8 hoặc 4
        project='models',
        name='pcb_high_res',
        device=0         # Nếu dùng GPU thì để 0, nếu CPU thì xóa dòng này (nhưng sẽ chậm)
    )

if __name__ == "__main__":
    train_model()