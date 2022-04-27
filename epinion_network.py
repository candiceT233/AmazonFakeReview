import os

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx  # version networkx=2.3
import seaborn as sns
from collections import Counter
import time

BASE_PATH = "./datasets"
OUT_PATH="./outputs"
# BASE_PATH = "/content/drive/MyDrive/cs579/CS579-Project2/datasets"
EPINIONS_FILENAME = "soc-sign-epinions.txt"
ORIGIN_COLUMNS = ['FromNodeId', 'ToNodeId', 'Sign']


def print_ten_sample(d, msg="print out check: "):
    keylist = list(d.keys())  # edge list
    out = keylist[0: 10]
    # printing result
    print(msg)
    for key in out:
        print(f"{key} : {d[key]}")


def import_data(filename):
    df = pd.read_csv(filename,
                     skiprows=4,
                     # nrows=5000,
                     delimiter='\t',
                     names=ORIGIN_COLUMNS)
    return df


def df_to_graph(df):
    edges = []

    for index, row in df.iterrows():
        n1 = row['FromNodeId']
        n2 = row['ToNodeId']
        edges.append((n1,n2))

    print("Adding nodes to graph ...")
    # add nodes to graph
    u_nodes, p_nodes = zip(*edges) # key = edge = (usernode, productnode)
    nodes = list(set(u_nodes + p_nodes))
    print(f"Users[{len(u_nodes)}] , Products[{len(p_nodes)}], Nodes[{len(nodes)}]")

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G


def compute_product_goodness_fairness(df):
    """
    product node:
    - Goodness: averaged overall sign from all users [-1, 1]
      If A(10 <+1>) ==> 10,B(5 <-1>) ==> -5, C(2 <+1>, 3 <-1> ==> -1).
      Then normalize the goodness.
    - Fairness: normalized number of users rated across the whole datasets [-1, 1]
      If 10 user node rate A product,20 user node rate B product
      Then A fairness10/20=0.5, B fairness=20/20=1

      If 8 user node rate C product, max-in-degree=20
      Then (8-10)/(20/2)=?
    """
    p_dict = {}
    graph_dict = df_to_dict(df)
    u_nodes, p_nodes = zip(*graph_dict.keys())
    max_sign = 0
    for node in p_nodes:
        # in-degree
        edges = G.in_edges(node)
        if max_sign < len(edges):
            max_sign = len(edges)
    print(max_sign)

    for node in p_nodes:
        edges = G.in_edges(node)
        # print(edges)
        sum_sign = 0
        for edge in edges:
            sum_sign += graph_dict[edge]['Sign']
        sum_sign = sum_sign / len(edges)
        fair = len(edges) / max_sign
        p_dict[node] = {"good": sum_sign, "fair": fair}

    # # print for verification
    # count = 0
    # for node in p_nodes:
    #     print(f"[{node}] goodness: {p_dict[node]['good']}; fairness: {p_dict[node]['fair']}")
    #     count += 1
    #     if count == 5:
    #         break

    return p_dict


def compute_user_goodness_fairness(df, p_dict):
    """
    user node:
      - Goodness:
        If A rate 10 product, value = 10
        If B rate 5 product, value = 5

    - Fairness:
      If A give P1 1

    """
    u_dict = {}
    graph_dict = df_to_dict(df)
    u_nodes, p_nodes = zip(*graph_dict.keys())
    for node in u_nodes:
        edges = G.out_edges(node)
        sum_good = len(edges)
        sum_fair = 0
        for edge in edges:
            sum_fair += (graph_dict[edge]['Sign'] - p_dict[edge[1]]['good'])
        sum_good = sum_good / len(edges)
        sum_fair = sum_fair / len(edges)
        u_dict[node] = {'good': sum_good, "fair": sum_fair}

    # # print for verification
    # count = 0
    # for node in u_nodes:
    #     print(f"[{node}] goodness: {u_dict[node]['good']}; fairness: {u_dict[node]['fair']}")
    #     count += 1
    #     if count == 5:
    #         break
    return u_dict


def compute_trust_dict(G):
    graph_dict = df_to_dict(df)
    trust_dict = {}
    for node in G.nodes:
        edges = G.in_edges(node)
        sum_rate = 0
        for edge in edges:
            sum_rate += graph_dict[edge]['Sign']
        if sum_rate < 0:
            trust_dict[node] = -1
        else:
            trust_dict[node] = 1
    return trust_dict


def plot_small_df(df):
    small_df = df.drop(df.index[1000:])

    # {(node1,node2):weight}
    small_nl = []
    edge_list = []

    for index, row in small_df.iterrows():
        n1 = row['FromNodeId']
        n2 = row['ToNodeId']
        edge = (n1, n2)
        edge_list.append(edge)

    # add nodes and edges to graph
    SG = nx.DiGraph()
    SG.add_nodes_from(small_nl)
    SG.add_edges_from(edge_list)

    node_lb = {}
    for e in SG.edges:
        n1 = e[0]
        n2 = e[1]
        node_lb[n2] = SG.in_degree(n2)
        node_lb[n1] = SG.in_degree(n1)

    plt.figure(figsize=(20, 10))

    pos = nx.drawing.nx_pydot.pydot_layout(SG, prog='fdp')
    # nx.draw(G, pos, node_color="#748AF9") # base color lightblue
    nx.draw(SG, pos)
    # nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
    nx.draw_networkx_labels(SG, pos, node_lb)

    plt.draw()
    plt.savefig(f'{OUT_PATH}/epinion_network_1000n.png')
    #plt.show()


# @title def plot_hist(G): plot degree histogram

## Plot histogram
def plot_hist(G):
  print("Plot degree histogram ...")
  plt.figure(figsize=(5, 5))
  deg_dict = Counter(nx.degree_histogram(G))

  y = []
  for key in deg_dict.keys():
    y.append(deg_dict[key])

  y = sorted(y,reverse=False)
  x = nx.degree_histogram(G)
  x = sorted(x,reverse=False)

  plt.hist(x, edgecolor="red", bins=y)
  plt.title("Degree Histogram")
  plt.ylabel("Count")
  plt.xlabel("Degree")
  plt.savefig(f'{OUT_PATH}/epinion_network_histogram.png')
  #plt.show()


# @title def plot_degree_dens(G): plot density of degree distribution
# https://seaborn.pydata.org/generated/seaborn.kdeplot.html
def plot_degree_dens(G):
  print("Plot degree density graph ...")
  plt.figure(figsize=(8, 4))
  x = list(G.nodes)
  y = []
  for node in x:
    y.append(G.degree(node))

  # Make density plot
  sns.kdeplot(y)
  plt.title("Degree Density Distribution")
  plt.ylabel("Degree Density")
  plt.xlabel("Degree")
  plt.xlim([0,1500])
  plt.savefig(f'{OUT_PATH}/epinion_network_degree_density.png')
  #plt.show()


# @title def get_astp(G): record average shortest path
# @markdown output: Nodes num: 55223, shortest path mean: 3.6516

# https://networkx.org/documentation/networkx-1.10/reference/generated/networkx.algorithms.shortest_paths.generic.average_shortest_path_length.html
def get_astp(G):
  print("Getting averaged shortest path ...")
  output_path=f'{OUT_PATH}/epinion_network_aveSTP_cluster.txt'

  length = nx.average_shortest_path_length(G)
  ave_clus = nx.average_clustering(G)
  with open(output_path, encoding='utf-8', mode='w+') as f:
      f.write(f'Averate Clustering Coefficients: {ave_clus}\n')
      f.write(f'Nodes num: {len(G.nodes())}, average shortest path: {length}\n')


if __name__ == "__main__":
    # import dataset
    # https://snap.stanford.edu/data/soc-sign-epinions.html
    filename = os.path.join(BASE_PATH, EPINIONS_FILENAME)

    start = time.time()
    df = import_data(filename)
    read_t = time.time()
    print(f"\nRead data time : {read_t - start}\n")

    #G = df_to_graph(df)
    graph_t = time.time()
    print(f"\nAdd to graph time : {graph_t - read_t}\n")

    # plot a small network structure sneakpeak
    plot_small_df(df)


    G = df_to_graph(df)
    graph_t = time.time()
    print(f"\nAdd to graph time : {graph_t - read_t}\n")

    # plot histogram
    plot_hist(G)

    """
    # realword network has:
    # (1) power law distribution (y)
    # (2) high clustering coefficients (n)
    # (3) shorts average path length (y)
    """
    # plot degree density
    plot_degree_dens(G)

    get_astp(G)
    end = time.time()
    print(f"\nTotal runtime : {end - start}\n")
    # """
