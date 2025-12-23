import cv2
import numpy as np

class TraceExtractor:
    def __init__(self):
        # Thông số Adaptive Threshold để bắt mạch sắc nét
        self.block_size = 31
        self.c_val = 3

    def process(self, image_path, components):
        # 1. Đọc ảnh và chuyển hệ xám
        img_bgr = cv2.imread(image_path)
        if img_bgr is None: return None, None, None
        
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Tăng cường độ tương phản (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # 2. Tạo Mask thô (Dạng Binary: Nền trắng, Mạch đen - giống ảnh 2 bạn gửi)
        # Sử dụng THRESH_BINARY thay vì THRESH_BINARY_INV
        mask_raw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, self.block_size, self.c_val)

        # 3. XÓA LINH KIỆN BẰNG MÀU TRẮNG (Theo yêu cầu của bạn)
        trace_only_white_bg = mask_raw.copy()
        for comp in components:
            x1, y1, x2, y2 = map(int, comp['box'])
            padding = 7
            # Vẽ màu trắng (255) đè lên linh kiện để xóa nó khỏi tầm mắt
            cv2.rectangle(trace_only_white_bg, (x1-padding, y1-padding), (x2+padding, y2+padding), 0, -1)

        # 4. LÀM SẠCH NHIỄU (Bởi vì mask raw của bạn đang bị rất nhiều chấm đen li ti)
        # Median Blur giúp xóa các chấm đen nhỏ trên nền trắng
        trace_only_white_bg = cv2.medianBlur(trace_only_white_bg, 3)
        kernel = np.ones((4, 4), np.uint8)
        trace_dilated = cv2.dilate(trace_only_white_bg, kernel, iterations=1)
        # 5. QUAN TRỌNG: Đảo ngược màu để Skeletonize
        # Chuyển từ (Nền trắng, mạch đen) -> (Nền đen, mạch trắng)
        mask_for_skeleton = cv2.bitwise_not(trace_only_white_bg)
        
        # Xóa bớt các hạt trắng nhỏ sinh ra sau khi đảo ngược
        kernel = np.ones((2,2), np.uint8)
        mask_for_skeleton = cv2.morphologyEx(mask_for_skeleton, cv2.MORPH_OPEN, kernel)

        # 6. SKELETON HÓA
        skeleton = self.skeletonize(mask_for_skeleton)

        # Trả về: Mask thô, Trace sạch (nền trắng), và Skeleton (nền đen mạch trắng)
        return mask_raw, trace_only_white_bg, skeleton

    def skeletonize(self, img):
        skel = np.zeros(img.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        temp_img = img.copy()
        while True:
            eroded = cv2.erode(temp_img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(temp_img, temp)
            skel = cv2.bitwise_or(skel, temp)
            temp_img = eroded.copy()
            if cv2.countNonZero(temp_img) == 0: break
        return skel
    def find_endpoints(self, skeleton):
        """Tìm các đường biên/tọa độ mạch sau khi đã làm mảnh"""
        contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours