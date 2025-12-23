import cv2
import matplotlib.pyplot as plt
import networkx as nx
from src.detector import ComponentDetector
from src.trace_extractor import TraceExtractor
from src.graph_builder import SchematicBuilder
import os
import sys

# Thêm đường dẫn hiện tại vào path để tránh lỗi import
sys.path.append(os.getcwd())

def main(image_path):
    # --- CẤU HÌNH ---
    model_path = r"PCB_Schematic_Reconstruction - Copy\PCB_Schematic_Reconstruction - Copy\models\pcb_high_res3\weights\best.pt"
    
    if not os.path.exists(model_path):
        print(f"Lỗi: Không tìm thấy file model tại {model_path}")
        print("Vui lòng kiểm tra lại đường dẫn file .pt")
        return

    if not os.path.exists(image_path):
        print(f"Lỗi: Không tìm thấy ảnh tại {image_path}")
        return

    # 1. Detect Components (Nhận diện linh kiện)
    print("Đang nhận diện linh kiện...")
    detector = ComponentDetector(model_path)
    components = detector.detect(image_path)
    print(f"Tìm thấy {len(components)} linh kiện.")

    # 2. Extract Traces (Xử lý đường mạch)
    print("Đang xử lý đường mạch...")
    trace_extractor = TraceExtractor()
    mask_hsv, mask_trace, skeleton = trace_extractor.process(image_path,components)
    contours = trace_extractor.find_endpoints(skeleton)

    # --- BƯỚC LỌC QUAN TRỌNG ---
    # Loại bỏ các class không mong muốn (như chân hàn, đinh ốc...) để sơ đồ sạch đẹp
    ignore_classes = ['pads', 'hole', 'mounting_hole', 'pins'] 
    
    clean_components = [
        comp for comp in components 
        if comp['class'] not in ignore_classes
    ]
    # ---------------------------

    # 3. Build Graph (Dùng danh sách linh kiện đã lọc)
    print("Đang tái tạo sơ đồ...")
    builder = SchematicBuilder()
    graph = builder.build_graph(clean_components, skeleton) # <-- Nhớ đổi biến ở đây

    # 4. Visualization (Hiển thị kết quả)
    
    # Đọc lại ảnh gốc để vẽ lên
    img = cv2.imread(image_path)
    # Chuyển từ BGR (OpenCV) sang RGB (Matplotlib)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Vẽ bounding box và tên linh kiện lên ảnh
    for comp in clean_components: # <-- Đổi biến ở đây
        x1, y1, x2, y2 = comp['box']
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img_rgb, comp['class'], (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # --- VẼ BIỂU ĐỒ ---
    plt.figure(figsize=(18, 8))

    # Cột 1 - Hàng 1: Ảnh gốc + Detection
    plt.subplot(2, 3, 1)
    plt.imshow(img_rgb)
    plt.title("1. Detection Result")
    plt.axis('off')

    # Cột 2 - Hàng 1: Mask HSV
    plt.subplot(2, 3, 2)
    plt.imshow(mask_hsv, cmap='gray')
    plt.title("2. adaptive Mask (Raw)")
    plt.axis('off')

    # Cột 3 - Hàng 1: Trace Only (Sau khi xóa linh kiện)
    plt.subplot(2, 3, 3)
    plt.imshow(mask_trace, cmap='gray')
    plt.title("3. Trace Only (Cleaned)")
    plt.axis('off')

    # Cột 1 - Hàng 2: Skeleton
    plt.subplot(2, 3, 4)
    plt.imshow(skeleton, cmap='gray')
    plt.title("4. Skeletonized")
    plt.axis('off')

    # Cột 2 & 3 - Hàng 2: Sơ đồ Graph (Chiếm 2 ô)
    ax_graph = plt.subplot(2, 3, (5, 6))
    pos = nx.get_node_attributes(graph, 'pos')
    if pos:
        # Lật trục Y để khớp với hệ tọa độ ảnh
        pos = {k: (v[0], -v[1]) for k, v in pos.items()}
        labels = nx.get_node_attributes(graph, 'label')
        nx.draw_networkx_nodes(graph, pos, node_size=200, node_color='orange', ax=ax_graph)
        nx.draw_networkx_edges(graph, pos, edge_color='blue', width=1, ax=ax_graph)
        # Vẽ nhãn lệch đi một chút
        pos_labels = {k: (v[0], v[1] + 20) for k, v in pos.items()}
        nx.draw_networkx_labels(graph, pos_labels, labels, font_size=7, ax=ax_graph)
    
    plt.title("5. Reconstructed Schematic Graph")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Thay đường dẫn ảnh test của bạn vào đây
    #test_image = r"G:/xu_li_anh/PCB_Schematic_Reconstruction - Copy - Copy - Copy-20251219T140956Z-3-001/PCB_Schematic_Reconstruction - Copy - Copy - Copy/data/test/images/PCBA_518.jpg"
    test_image =r"PCB_Schematic_Reconstruction - Copy\pcb_xanh.jpg"
    # Kiểm tra xem file có tồn tại không trước khi chạy
    if os.path.exists(test_image):
        main(test_image)
    else:
        print(f"File ảnh không tồn tại: {test_image}")
        print("Hãy sửa lại đường dẫn trong biến test_image")