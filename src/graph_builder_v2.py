import networkx as nx
import cv2
import numpy as np

class SchematicBuilder:
    def __init__(self):
        self.graph = nx.Graph()

    # =========================
    # CHECK TOUCHING
    # =========================
    def is_touching(self, component_box, point, buffer=12):
        x, y = point
        x1, y1, x2, y2 = component_box
        return (x1-buffer <= x <= x2+buffer and
                y1-buffer <= y <= y2+buffer)

    # =========================
    # BUILD GRAPH
    # =========================
    def build_graph(self, components, skeleton):
        self.graph.clear()

        # 1. Add component nodes
        for i, comp in enumerate(components):
            self.graph.add_node(
                f"C{i}",
                type="component",
                label=comp["class"],
                pos=tuple(comp["center"]),
                box=comp["box"]
            )

        # 2. Find nets from skeleton
        num_labels, labels = cv2.connectedComponents(
            (skeleton > 0).astype(np.uint8)
        )

        # 3. For each net
        for net_id in range(1, num_labels):
            net_mask = (labels == net_id)
            net_points = np.column_stack(np.where(net_mask))
            net_points = net_points[:, [1, 0]]  # (x, y)

            # Create net node
            net_node = f"N{net_id}"
            self.graph.add_node(
                net_node,
                type="net",
                points=len(net_points)
            )

            touched_components = set()

            # 4. Check touching using ALL points near components
            for i, comp in enumerate(components):
                box = comp["box"]

                # Lấy các điểm skeleton nằm gần bbox
                for pt in net_points:
                    if self.is_touching(box, pt):
                        touched_components.add(f"C{i}")
                        break   # chỉ cần 1 điểm là đủ

            # 5. Connect net to components
            for comp_node in touched_components:
                self.graph.add_edge(net_node, comp_node)

        return self.graph