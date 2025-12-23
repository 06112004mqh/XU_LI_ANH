import cv2
import matplotlib.pyplot as plt
import networkx as nx

def visualize_results(image, graph, components):
    # Vẽ Bounding Box
    vis_img = image.copy()
    for comp in components:
        x1, y1, x2, y2 = comp['bbox']
        color = (0, 255, 0) # Xanh lá
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis_img, comp['id'], (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Vẽ sơ đồ kết nối (NetworkX plot)
    plt.figure(figsize=(12, 6))
    
    # Subplot 1: Ảnh gốc + BBox
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.title("Detected Components")
    plt.axis('off')
    
    # Subplot 2: Graph
    plt.subplot(1, 2, 2)
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color='skyblue', 
            node_size=1500, font_size=10, font_weight='bold')
    plt.title("Reconstructed Schematic Graph")
    
    plt.tight_layout()
    plt.show()