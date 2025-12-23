import networkx as nx
import cv2
import numpy as np

class SchematicBuilder:
    def __init__(self):
        # Đồ thị sẽ được khởi tạo mới mỗi khi build
        pass

    def is_touching(self, component_box, trace_point, buffer=12):
        """
        Kiểm tra điểm mạch có nằm trong vùng lân cận của linh kiện không.
        buffer: Nên lớn hơn hoặc bằng padding mà bạn dùng khi xóa linh kiện.
        """
        x, y = trace_point
        x1, y1, x2, y2 = component_box
        # Kiểm tra xem điểm có nằm trong hộp linh kiện đã mở rộng (buffer) hay không
        return (x1 - buffer <= x <= x2 + buffer) and (y1 - buffer <= y <= y2 + buffer)

    def build_graph(self, components, skeleton):
        """
        components: list các linh kiện từ detector [{'class': 'resistor', 'box': [x1,y1,x2,y2], 'center': [x,y]}]
        skeleton: Ảnh 1-pixel skeleton đã qua xử lý nối mạch (final_skeleton)
        """
        self.graph = nx.Graph()
        
        # 1. Thêm các linh kiện làm Nodes
        for i, comp in enumerate(components):
            self.graph.add_node(i, 
                               label=comp['class'], 
                               pos=comp['center'],
                               box=comp['box'])

        # 2. Phân tách skeleton thành các "Net" (Các đường nối rời rạc)
        # Mỗi cụm điểm trắng liên tục trong skeleton được coi là một dây (Net)
        num_labels, labels_im = cv2.connectedComponents(skeleton)

        # 3. Duyệt qua từng Net (nhãn 0 là nền đen, bỏ qua)
        for label in range(1, num_labels):
            net_points = np.argwhere(labels_im == label)
            # Chuyển từ (y, x) của numpy sang (x, y) của tọa độ ảnh
            net_points = net_points[:, [1, 0]] 
            
            # Tập hợp các linh kiện mà "Net" này chạm vào
            touched_components = set()

            # Để tối ưu, chúng ta lấy mẫu điểm trên Net thay vì kiểm tra toàn bộ
            # Cứ mỗi 5 pixel lấy 1 điểm để check va chạm
            sampling_rate = 15
            sampled_points = net_points[::sampling_rate]
            
            # Thêm các điểm đầu và điểm cuối của dây (End-points) vì chúng dễ chạm linh kiện nhất
            sampled_points = np.vstack([sampled_points, net_points[0], net_points[-1]])

            for pt in sampled_points:
                pt_x, pt_y = pt
                for i, comp in enumerate(components):
                    if self.is_touching(comp['box'], (pt_x, pt_y)):
                        touched_components.add(i)
            
            # 4. Nếu 1 Net chạm vào >= 2 linh kiện, tạo cạnh nối chúng
            # Ví dụ: Nếu Net chạm vào R1, C2 và IC1 -> tạo cạnh (R1-C2), (C2-IC1)
            connected_list = list(touched_components)
            if len(connected_list) >= 2:
                # Tạo kết nối chuỗi (u-v-w) hoặc kết nối đầy đủ (clique)
                for k in range(len(connected_list)):
                    for m in range(k + 1, len(connected_list)):
                        u = connected_list[k]
                        v = connected_list[m]
                        # Thêm cạnh với thông tin là label của Net đó
                        self.graph.add_edge(u, v, net_id=label)

        return self.graph