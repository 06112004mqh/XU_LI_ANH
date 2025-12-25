import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx

# Setup đường dẫn
sys.path.append(os.getcwd())

# ==========================================
# IMPORT MODULES TỪ CODE B (V2)
# ==========================================
try:
    from src.detector import ComponentDetector
    from src.trace_extractor_v2 import TraceExtractor
    from src.graph_builder_v2 import SchematicBuilder
except ImportError as e:
    print(f"Lỗi Import: {e}. Hãy đảm bảo cấu trúc thư mục src đúng.")

# ==========================================
# 0. CẤU HÌNH UI & CONSTANTS
# ==========================================
IGNORE_CLASSES = ['pads', 'hole', 'mounting_hole', 'pins']

# ==========================================
# 1. HÀM XỬ LÝ ẢNH (GIỮ TỪ CODE A)
# ==========================================
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] 
    rect[2] = pts[np.argmax(s)] 
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] 
    rect[3] = pts[np.argmax(diff)] 
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0],[maxWidth - 1, 0],[maxWidth - 1, maxHeight - 1],[0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def auto_correct_perspective(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    if not cnts: return image
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break
    if screenCnt is None: return image
    return four_point_transform(image, screenCnt.reshape(4, 2))

def apply_image_filters(cv_img, brightness=0, contrast=1.0, denoise=0):
    img_res = cv2.convertScaleAbs(cv_img, alpha=contrast, beta=brightness)
    if denoise > 0:
        ksize = int(denoise) * 2 + 1 
        img_res = cv2.GaussianBlur(img_res, (ksize, ksize), 0)
    return img_res

# ==========================================
# 2. GUI APPLICATION (MAIN)
# ==========================================
ctk.set_appearance_mode("Dark") 
ctk.set_default_color_theme("blue")

class SchematicApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("PCB Analysis & Reconstruction")
        self.geometry("1300x850") # Rộng hơn chút để hiển thị đồ thị
        self.minsize(1000, 700)
        
        # --- CẤU HÌNH ĐƯỜNG DẪN MODEL ---
        # Sửa lại đường dẫn này cho đúng với máy của bạn
        self.model_path = r"G:\xu_li_anh\PCB_Schematic_Reconstruction - Copy - Copy - Copy-20251219T140956Z-3-001\PCB_Schematic_Reconstruction - Copy - Copy - Copy\models\pcb_high_res3\weights\best.pt"
        
        self.original_cv_image = None
        self.processed_cv_image = None
        self.image_path_loaded = None # Lưu path gốc nếu cần

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # --- SIDEBAR ---
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        self.logo_label = ctk.CTkLabel(self.sidebar, text="PCB TO GRAPH", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.pack(padx=20, pady=(20, 10))
        
        self.btn_load = ctk.CTkButton(self.sidebar, text="📂 Load Image", font=ctk.CTkFont(size=14), height=40, command=self.load_image)
        self.btn_load.pack(padx=20, pady=20)
        
        self.lbl_filters = ctk.CTkLabel(self.sidebar, text="Preprocessing", font=ctk.CTkFont(size=14, weight="bold"), anchor="w")
        self.lbl_filters.pack(padx=20, pady=(10, 10), fill="x")

        # NÚT AUTO CALIB
        self.btn_calib = ctk.CTkButton(self.sidebar, text="📐 Auto Calib (Deskew)", font=ctk.CTkFont(size=13), height=30, fg_color="#D35400", hover_color="#A04000", command=self.action_auto_calib)
        self.btn_calib.pack(padx=20, pady=(0, 20), fill="x")
        
        ctk.CTkLabel(self.sidebar, text="Brightness").pack(padx=20, anchor="w")
        self.slider_bright = ctk.CTkSlider(self.sidebar, from_=-100, to=100, number_of_steps=20, command=self.update_preview)
        self.slider_bright.set(0)
        self.slider_bright.pack(padx=20, pady=(0, 15), fill="x")
        
        ctk.CTkLabel(self.sidebar, text="Contrast").pack(padx=20, anchor="w")
        self.slider_contrast = ctk.CTkSlider(self.sidebar, from_=0.5, to=3.0, number_of_steps=25, command=self.update_preview)
        self.slider_contrast.set(1.0)
        self.slider_contrast.pack(padx=20, pady=(0, 15), fill="x")
        
        ctk.CTkLabel(self.sidebar, text="Denoise").pack(padx=20, anchor="w")
        self.slider_denoise = ctk.CTkSlider(self.sidebar, from_=0, to=5, number_of_steps=5, command=self.update_preview)
        self.slider_denoise.set(0)
        self.slider_denoise.pack(padx=20, pady=(0, 20), fill="x")
        
        self.btn_reset = ctk.CTkButton(self.sidebar, text="Reset Filters", fg_color="transparent", border_width=1, command=self.reset_filters)
        self.btn_reset.pack(padx=20, pady=5)
        
        self.btn_generate = ctk.CTkButton(self.sidebar, text="⚡ ANALYZE PCB", 
                                          font=ctk.CTkFont(size=16, weight="bold"), 
                                          fg_color="#00AA00", hover_color="#008800", height=60,
                                          state="disabled", command=self.run_analysis)
        self.btn_generate.pack(padx=20, pady=(40, 20), side="bottom")

        # --- MAIN AREA ---
        self.tabview = ctk.CTkTabview(self, corner_radius=10)
        self.tabview.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        self.tab_preview = self.tabview.add("Preview & Detect")
        self.tab_result = self.tabview.add("Analysis Result")
        
        self.lbl_preview_img = ctk.CTkLabel(self.tab_preview, text="Vui lòng chọn ảnh PCB...", corner_radius=10)
        self.lbl_preview_img.pack(expand=True, fill="both", padx=10, pady=10)

        # Frame chứa Matplotlib result
        self.result_frame = ctk.CTkFrame(self.tab_result)
        self.result_frame.pack(fill="both", expand=True)

    # --- ACTION HANDLERS ---
    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg;*.png;*.jpeg")])
        if path:
            self.image_path_loaded = path
            self.original_cv_image = cv2.imread(path)
            self.original_cv_image = cv2.cvtColor(self.original_cv_image, cv2.COLOR_BGR2RGB)
            self.reset_filters() 
            self.btn_generate.configure(state="normal")
            self.tabview.set("Preview & Detect")

    def reset_filters(self):
        self.slider_bright.set(0)
        self.slider_contrast.set(1.0)
        self.slider_denoise.set(0)
        self.update_preview()

    def display_pil_image(self, img_pil, label_widget):
        w_win, h_win = self.tabview.winfo_width(), self.tabview.winfo_height()
        if w_win < 100: w_win, h_win = 800, 600
        
        ratio = min((w_win-50)/img_pil.width, (h_win-50)/img_pil.height)
        new_w, new_h = int(img_pil.width * ratio), int(img_pil.height * ratio)
        
        if new_w > 0 and new_h > 0:
            img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
        ctk_img = ctk.CTkImage(img_pil, size=(new_w, new_h))
        label_widget.configure(image=ctk_img, text="")

    def action_auto_calib(self):
        if self.original_cv_image is None: return
        calib_img = auto_correct_perspective(self.original_cv_image)
        self.original_cv_image = calib_img
        self.update_preview()
        messagebox.showinfo("Auto Calib", "Đã căn chỉnh góc ảnh!")

    def update_preview(self, val=None):
        if self.original_cv_image is None: return
        b = int(self.slider_bright.get())
        c = self.slider_contrast.get()
        d = int(self.slider_denoise.get())
        self.processed_cv_image = apply_image_filters(self.original_cv_image, b, c, d)
        
        img_pil = Image.fromarray(self.processed_cv_image)
        self.display_pil_image(img_pil, self.lbl_preview_img)

    # --- LOGIC PHÂN TÍCH (TỪ CODE B) ---
    def run_analysis(self):
        if self.processed_cv_image is None: return
        
        if not os.path.exists(self.model_path):
            messagebox.showerror("Lỗi", f"Không tìm thấy model tại:\n{self.model_path}")
            return

        # 1. Lưu ảnh tạm để xử lý (Apply các filter người dùng đã chỉnh)
        temp_path = "temp_analysis.jpg"
        cv2.imwrite(temp_path, cv2.cvtColor(self.processed_cv_image, cv2.COLOR_RGB2BGR))
        
        try:
            print("1. Detecting components...")
            detector = ComponentDetector(self.model_path)
            components = detector.detect(temp_path)
            print(f"   Found {len(components)} components")

            print("2. Extracting traces...")
            extractor = TraceExtractor()
            # Code B trả về 3 biến
            mask, skeleton, skeleton_connected = extractor.process(temp_path, components)

            # 3. Filter components
            clean_components = [c for c in components if c['class'] not in IGNORE_CLASSES]

            # 4. Build schematic graph
            print("3. Building graph...")
            builder = SchematicBuilder()
            graph = builder.build_graph(clean_components, skeleton_connected)

            # --- VISUALIZATION (Tạo Figure Matplotlib) ---
            print("4. Rendering results...")
            
            # Xóa các widget cũ trong tab kết quả
            for widget in self.result_frame.winfo_children():
                widget.destroy()

            # Tạo Figure lớn
            fig = plt.figure(figsize=(14, 8), dpi=100)
            
            # Subplot 1: Ảnh gốc + Bounding Boxes
            img_rgb_vis = self.processed_cv_image.copy()
            for comp in clean_components:
                x1, y1, x2, y2 = comp['box']
                cv2.rectangle(img_rgb_vis, (x1, y1), (x2, y2), (255,0,0), 3)
                cv2.putText(img_rgb_vis, comp['class'], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
            
            ax1 = fig.add_subplot(2, 3, 1)
            ax1.imshow(img_rgb_vis)
            ax1.set_title("1. Detection")
            ax1.axis('off')

            # Subplot 2: Trace Mask
            ax2 = fig.add_subplot(2, 3, 2)
            ax2.imshow(mask, cmap='gray')
            ax2.set_title("2. Trace Mask")
            ax2.axis('off')

            # Subplot 3: Skeleton
            ax3 = fig.add_subplot(2, 3, 3)
            ax3.imshow(skeleton, cmap='gray')
            ax3.set_title("3. Skeleton")
            ax3.axis('off')

            # Subplot 4: Connected Skeleton
            ax4 = fig.add_subplot(2, 3, 4)
            ax4.imshow(skeleton_connected, cmap='gray')
            ax4.set_title("4. Skeleton (Connected)")
            ax4.axis('off')

            # Subplot 5: Component Graph (Logic từ Code B)
            ax5 = fig.add_subplot(2, 3, (5, 6))
            
            comp_graph = nx.Graph()
            # Thêm node Component
            for node, data in graph.nodes(data=True):
                if data.get("type") == "component":
                    comp_graph.add_node(node, **data)
            
            # Nối Component thông qua Net
            net_nodes = [n for n, d in graph.nodes(data=True) if d.get("type") == "net"]
            for net in net_nodes:
                neighbors = list(graph.neighbors(net))
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        u, v = neighbors[i], neighbors[j]
                        if comp_graph.has_node(u) and comp_graph.has_node(v):
                            comp_graph.add_edge(u, v)
            
            # Vẽ Graph
            if comp_graph.number_of_nodes() > 0:
                pos = {node: (data["pos"][0], -data["pos"][1]) for node, data in comp_graph.nodes(data=True)}
                labels = {node: data.get("label", node) for node, data in comp_graph.nodes(data=True)}
                
                nx.draw_networkx_nodes(comp_graph, pos, node_size=300, node_color="orange", edgecolors="black", ax=ax5)
                nx.draw_networkx_edges(comp_graph, pos, width=1.5, edge_color='blue', alpha=0.5, ax=ax5)
                nx.draw_networkx_labels(comp_graph, pos, labels, font_size=8, font_weight='bold', ax=ax5)
                ax5.set_title("5. Component Connectivity Graph")
                ax5.axis("off")
            else:
                ax5.text(0.5, 0.5, "No components connected", ha='center', va='center')

            plt.tight_layout()

            # --- NHÚNG FIGURE VÀO TKINTER UI ---
            canvas = FigureCanvasTkAgg(fig, master=self.result_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            # Chuyển sang tab kết quả
            self.tabview.set("Analysis Result")
            
            # Xóa ảnh tạm
            if os.path.exists(temp_path):
                os.remove(temp_path)

        except Exception as e:
            messagebox.showerror("Lỗi", f"Có lỗi xảy ra: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    app = SchematicApp()
    app.mainloop()