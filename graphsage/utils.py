from __future__ import print_function

import numpy as np
import random
import json
import sys
import os

import networkx as nx
from networkx.readwrite import json_graph
version_info = list(map(int, nx.__version__.split('.')))
major = version_info[0]
minor = version_info[1]
assert (major <= 1) and (minor <= 11), "networkx major version > 1.11"

WALK_LEN=5 
N_WALKS=50

def load_data(prefix, normalize=True, load_walks=False):
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)  #将json格式的图转换为networkx图
    if isinstance(G.nodes()[0], int):   #判断图的节点是否为整数
        conversion = lambda n : int(n)  #如果是整数，转换为整数
    else:
        conversion = lambda n : n   #如果不是整数，不转换

    #加载节点特征， ppi有50个特征， 它的形状是56944*50， 特征是稀疏的1，0
    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None

    #加载节点id_map
    id_map = json.load(open(prefix + "-id_map.json"))
    id_map = {conversion(k):int(v) for k,v in id_map.items()}   #将id_map的key转换为整数，value转换为整数
    walks = []  #当前为空，后面会加载随机游走的数据
    class_map = json.load(open(prefix + "-class_map.json")) #加载标签数据集，就是一个节点对应121个标签的数据集
    if isinstance(list(class_map.values())[0], list):   #判断标签是否为列表，一般来说都是列表
        lab_conversion = lambda n : n
    else:
        lab_conversion = lambda n : int(n)

    # 将标签数据集的key转换为整数，value转换为整数
    class_map = {conversion(k):lab_conversion(v) for k,v in class_map.items()}

    ## Remove all nodes that do not have val/test annotations
    ## (necessary because of networkx weirdness with the Reddit data)
    broken_count = 0    #记录删除的节点数
    for node in G.nodes():
        if not 'val' in G.node[node] or not 'test' in G.node[node]:
            G.remove_node(node)
            broken_count += 1
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")

    #遍历图G的所有边，然后根据边连接的节点的属性’val’和‘’test‘’来设置边的属性‘train_removed‘， 决定是否从训练集中删除
    #第一个循环确保val and test不会进入训练集，G.node[edge[0]]代表当前边的第一个节点，G.node[edge[1]]代表当前边的第二个节点
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
            G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    #如果normalize为True，且特征不为空，则对特征进行标准化
    #说一下标准化， feats是一个56944*50的矩阵，每一行代表一个节点的特征，每一列代表一个特征
    #标准化计算了每个特征（就是每一列）的均值和标准差，然后对每个特征减去均值，然后除以标准差得倒标准化后的特征
    #这样做的目的是为了让每个特征的均值为0，方差为1，确保没有一个特征因为其数值范围的大小而对训练的结果产生影响，增强泛化能力
    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        #选择训练集的特征
        #创建一个新数组train_ids，存储训练集的节点id
        train_ids = np.array([id_map[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
        train_feats = feats[train_ids]  #根据训练集的节点id，选择训练集的特征
        scaler = StandardScaler()      #创建一个标准化的对象
        scaler.fit(train_feats)     #对训练集的特征进行标准化， fit计算出每个特征的均值和标准差
        feats = scaler.transform(feats)   #trasnform里用公式根据均值和标准差对特征进行计算
    
    #如果load_walks为True，加载随机游走的数据
    if load_walks:
        with open(prefix + "-walks.txt") as fp: #打开文件并重命名为fp
            for line in fp:
                #walk是之前定义的空列表，map是python内置函数，将line按空格分割，然后转换为整数添加到walk，数据不变
                walks.append(map(conversion, line.split()))

    #最后data_loader返回图G，标准化后的特征feats，id_map，walks，class_map，后面三个基本没变
    return G, feats, id_map, walks, class_map

#随机游走函数
def run_random_walks(G, nodes, num_walks=N_WALKS):  #G是连接图，nodes是存储节点的列表 ，N_WALKS=50
    pairs = []  #定义一个空列表，用来存储随机游走的节点对
    for count, node in enumerate(nodes):
        if G.degree(node) == 0:  #如果节点的度为0，跳过。degree返回节点的边数（对于无向图）
            continue
        for i in range(num_walks):  #对于每个节点，进行50次随机游走
            curr_node = node
            for j in range(WALK_LEN):   #每次随机游走5步
                next_node = random.choice(G.neighbors(curr_node)) #从当前节点的邻居中随机选择一个节点
                # self co-occurrences are useless
                if curr_node != node:
                    pairs.append((node,curr_node))  #将当前节点和下一个节点添加到pairs中
                curr_node = next_node  #当前节点更新为下一个节点
        if count % 1000 == 0:   #每1000个节点打印一次   
            print("Done walks for", count, "nodes")
    return pairs #存储了所有节点对的列表

if __name__ == "__main__":
    """ Run random walks """
    #sys.argv是一个列表，里面存储了命令行参数,sys.argv[1]和sys.argv[2]分别是第一个和第二个参数，
    #代表输入和输出文件的路径。
    graph_file = sys.argv[1]
    out_file = sys.argv[2]
    G_data = json.load(open(graph_file))
    G = json_graph.node_link_graph(G_data)
    #在这里定义了nodes,存储训练集
    nodes = [n for n in G.nodes() if not G.node[n]["val"] and not G.node[n]["test"]]
    #这个是只有训练节点的图
    G = G.subgraph(nodes)
    pairs = run_random_walks(G, nodes) 
    #将所有节点对写入文件，str(p[0]) + "\t" + str(p[1])是将节点对转换为字符串(才能写入文件)，然后用制表符分隔
    with open(out_file, "w") as fp:
        fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs]))
