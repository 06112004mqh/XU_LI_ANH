import cv2
import numpy as np

class TraceExtractor:
    def __init__(self):
        self.max_connect_dist = 5   # khoảng nối tối đa giữa 2 endpoint
        self.angle_cos_th = -0.7     # cos(góc) để xét đối hướng

    # =========================
    # MAIN PIPELINE
    # =========================
    def process(self, image_path, components):
        img = cv2.imread(image_path)
        if img is None:
            return None, None, None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Tăng tương phản
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        # 2. Threshold ổn định hơn adaptive
        _, mask = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Trace là màu tối → đảo
        mask = cv2.bitwise_not(mask)

        # 3. Xóa linh kiện
        for comp in components:
            x1, y1, x2, y2 = map(int, comp['box'])
            pad = 6
            cv2.rectangle(
                mask,
                (x1-pad, y1-pad),
                (x2+pad, y2+pad),
                0, -1
            )

        # 4. Làm sạch nhiễu nhỏ (KHÔNG dilate mạnh)
        kernel = np.ones((2,2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 5. Skeleton
        skeleton = self.skeletonize(mask)

        # 6. Tìm endpoint
        endpoints = self.find_endpoints(skeleton)

        # 7. Nối endpoint cùng hướng
        skeleton_connected = self.connect_endpoints(
            skeleton, endpoints
        )

        return mask, skeleton, skeleton_connected

    # =========================
    # SKELETONIZE
    # =========================
    def skeletonize(self, img):
        skel = np.zeros(img.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        temp = img.copy()

        while True:
            eroded = cv2.erode(temp, element)
            opened = cv2.dilate(eroded, element)
            residue = cv2.subtract(temp, opened)
            skel = cv2.bitwise_or(skel, residue)
            temp = eroded.copy()
            if cv2.countNonZero(temp) == 0:
                break
        return skel

    # =========================
    # ENDPOINT DETECTION
    # =========================
    def find_endpoints(self, skel):
        skel = (skel > 0).astype(np.uint8)
        endpoints = []

        h, w = skel.shape
        for y in range(1, h-1):
            for x in range(1, w-1):
                if skel[y, x]:
                    neighbors = np.sum(skel[y-1:y+2, x-1:x+2]) - 1
                    if neighbors == 1:
                        endpoints.append((x, y))
        return endpoints

    # =========================
    # LOCAL DIRECTION
    # =========================
    def get_direction(self, ep, skel):
        x, y = ep
        for dy in [-1,0,1]:
            for dx in [-1,0,1]:
                if dx == 0 and dy == 0:
                    continue
                ny, nx = y+dy, x+dx
                if skel[ny, nx]:
                    return np.array([dx, dy], dtype=np.float32)
        return None

    # =========================
    # CONNECT ENDPOINTS
    # =========================
    def connect_endpoints(self, skel, endpoints):
        skel = skel.copy()

        for i, ep1 in enumerate(endpoints):
            d1 = self.get_direction(ep1, skel)
            if d1 is None:
                continue

            for ep2 in endpoints[i+1:]:
                d2 = self.get_direction(ep2, skel)
                if d2 is None:
                    continue

                dist = np.linalg.norm(
                    np.array(ep1) - np.array(ep2)
                )
                if dist > self.max_connect_dist:
                    continue

                # kiểm tra cùng hướng (đối hướng)
                cos = np.dot(d1, d2) / (
                    np.linalg.norm(d1) * np.linalg.norm(d2)
                )

                if cos < self.angle_cos_th:
                    cv2.line(skel, ep1, ep2, 255, 1)

        return skel