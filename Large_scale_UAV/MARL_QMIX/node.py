import networkx as nx
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg


def nodes_location(nodes_num, distance=None):  # 生成环境节点和干扰机坐标
    # 接收机节点坐标
    receive_node_coordinate = {}
    for i in range(nodes_num):
        com_max_distance = 5  # km
        theta = random.uniform(0, 2 * math.pi)
        r = com_max_distance * random.random()

        x = round(r * math.cos(theta), 2)
        y = round(r * math.sin(theta), 2)
        receive_node_coordinate[i] = (x, y)

    # 干扰机坐标
    jam_nodes_coordinate = {}
    for j in range(2):
        theta = random.uniform(0, 2 * math.pi)
        if distance is not None:
            r = distance[j]
        else:
            r = random.uniform(160, 250)

        # 转换为笛卡尔坐标
        x = round(r * math.cos(theta), 2)
        y = round(r * math.sin(theta), 2)

        jam_nodes_coordinate[j] = (x, y)

    return receive_node_coordinate, jam_nodes_coordinate


def calculate_lbs(distance):
    """Compute free-space like path loss terms (vector on preset freqs).

    Guard against accidental shadowing of the global `np` by importing numpy
    locally. This fixes rare runtime errors where `np` or intermediate values
    become a numpy.ufunc due to external code re-binding names.
    """
    import numpy as _np  # local import to avoid any accidental shadowing

    freqs = 600.0 + 1.95 * _np.arange(40, dtype=float)
    # ensure scalar distance and numerical stability
    d = float(distance) if distance is not None else 0.0
    base = 32.45 + 20.0 * _np.log10(max(d, 1e-12) + 1.0)
    lbs = base + 20.0 * _np.log10(freqs)
    return _np.asarray(lbs, dtype=float)


def init_G(nodes_num=None, node_data=None, jam_distance_1=None, jam_distance_2=None, diasplay=False):
    node_num = nodes_num
    jam_distance = None
    if jam_distance_1 is not None:
        jam_distance = [jam_distance_1, jam_distance_2]

    G = nx.complete_graph(node_num)
    pos_node, pos_jam = nodes_location(node_num, distance=jam_distance)

    # 支持外部传入3D坐标（x,y,z）；仍保存二维pos用于兼容现有状态与绘图
    if node_data is not None:
        for i in range(node_num):
            pdata = node_data[i]
            if isinstance(pdata, (list, tuple)) and len(pdata) >= 3:
                pos_node[i] = (pdata[0], pdata[1])
            else:
                pos_node[i] = pdata

    new_node_id1 = len(G.nodes)
    new_node_id2 = len(G.nodes) + 1
    G.add_node(new_node_id1)
    G.add_node(new_node_id2)
    pos_node[new_node_id1] = pos_jam[0]  # 为新节点指定坐标
    pos_node[new_node_id2] = pos_jam[1]
    # 与现有接收机节点完全连边，避免在 NodeView 迭代时动态修改造成的不确定性
    for existing_node in range(node_num):
        G.add_edge(new_node_id1, existing_node)
        G.add_edge(new_node_id2, existing_node)

    for node, coordinates in pos_node.items():
        G.nodes[node]['pos'] = coordinates
    # 如果传入了3D坐标，保存到节点属性 pos3d（z默认0）
    if node_data is not None:
        for i in range(node_num):
            pdata = node_data[i]
            if isinstance(pdata, (list, tuple)) and len(pdata) >= 3:
                G.nodes[i]['pos3d'] = (float(pdata[0]), float(pdata[1]), float(pdata[2]))
    # 为干扰机设置3D坐标（z=0）
    G.nodes[new_node_id1]['pos3d'] = (float(pos_node[new_node_id1][0]), float(pos_node[new_node_id1][1]), 0.0)
    G.nodes[new_node_id2]['pos3d'] = (float(pos_node[new_node_id2][0]), float(pos_node[new_node_id2][1]), 0.0)

    # 距离按3D计算（若可用），否则回退到2D
    for (u, v) in G.edges():
        x1, y1 = pos_node[u][0], pos_node[u][1]
        x2, y2 = pos_node[v][0], pos_node[v][1]
        z1 = 0.0
        z2 = 0.0
        # 加强健壮性：某些场景下 G.nodes[u] 可能不是dict（例如被外部代码覆盖），
        # 这里做类型守卫以避免对tuple等类型使用字符串键索引导致异常。
        try:
            attrs_u = G.nodes[u]
        except Exception:
            attrs_u = {}
        try:
            attrs_v = G.nodes[v]
        except Exception:
            attrs_v = {}

        if isinstance(attrs_u, dict):
            pos3d_u = attrs_u.get('pos3d')
            if isinstance(pos3d_u, (list, tuple)) and len(pos3d_u) >= 3:
                z1 = float(pos3d_u[2])
        if isinstance(attrs_v, dict):
            pos3d_v = attrs_v.get('pos3d')
            if isinstance(pos3d_v, (list, tuple)) and len(pos3d_v) >= 3:
                z2 = float(pos3d_v[2])

        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
        G.edges[u, v]['distance'] = distance
        G.edges[v, u]['lbs'] = calculate_lbs(distance)

    if diasplay:
        background_img = mpimg.imread('C:/Users/admin/Desktop/waveform_design_aj/3.0/imgs/background.png')
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(background_img, extent=[-30, 30, -30, 30], aspect='auto')  # 调整extent来适应图片位置
        # plt.figure(figsize=(10, 10))
        nodes = [0, 1, 2, 3]
        jammer_nodes = [4, 5]
        # node_colors = ['lightblue' if node in selected_nodes else 'red' for node in G.nodes()]
        node_sizes = [500 if node in nodes else 1000 for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos_node, node_color='none', node_size=node_sizes)

        edges = [(u, v) for u, v in G.edges() if u in nodes and v in nodes]

        nx.draw_networkx_edges(G, pos_node, edgelist=edges, edge_color='red', width=3)
        nx.draw_networkx_labels(G, pos_node, font_size=12, font_color='red')

        node_image = mpimg.imread('C:/Users/admin/Desktop/waveform_design_aj/3.0/imgs/node.png')  # 替换为你本地的图片路径
        jammer_image = mpimg.imread('C:/Users/admin/Desktop/waveform_design_aj/3.0/imgs/jammer.png')  # 替换为你本地的图片路径
        for node in nodes:
            # 获取节点位置
            x, y = pos_node[node]
            # 创建图片对象
            imagebox = OffsetImage(node_image, zoom=0.2)  # 调整 zoom 来控制节点图片大小
            ab = AnnotationBbox(imagebox, (x, y), frameon=False)
            # 添加图片到图中
            ax.add_artist(ab)
        for node in jammer_nodes:
            # 获取节点位置
            x, y = pos_node[node]
            # 创建图片对象
            imagebox = OffsetImage(jammer_image, zoom=0.3)  # 调整 zoom 来控制节点图片大小
            ab = AnnotationBbox(imagebox, (x, y), frameon=False)
            # 添加图片到图中
            ax.add_artist(ab)

        plt.axis('off')
        plt.savefig('node.png', bbox_inches='tight')
        # plt.show()

    return G

# def draw_nx(G):

# if __name__ == '__main__':
#     G = init_G()
#     pass
# plt.figure(figsize=(6, 6))
# nx.draw(G, pos_node, with_labels=True, node_color='lightblue', node_size=500, font_size=16, font_weight='bold')
#
# # 在每条边上标注距离
# edge_labels = {(u, v): f"{d['distance']:.2f}" for u, v, d in G.edges(data=True)}
# nx.draw_networkx_edge_labels(G, pos_node, edge_labels=edge_labels, font_size=12)
#
# plt.title("Complete Graph with Distance Labels")
# plt.show()
# plt.close()
