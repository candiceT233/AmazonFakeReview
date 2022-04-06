import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter
import networkx as nx # version networkx=2.3
import seaborn as sns
from collections import Counter
from sklearn import preprocessing
import math
from statistics import mean
import random
import time

# Video Game product reviews
amazon_data="datasets/Video_Games_5.json"
rate_feat = ["overall", "verified", "reviewerID", "asin", "vote"]
review_feat = ["reviewerID", "asin", "reviewText", "summary"]
all_feat = ["reviewerID", "asin", "overall", "verified", "vote", "reviewText", "summary"]

def print_ten_sample(d, msg="print out check: "):
  keylist = list(d.keys()) # edge list
  out = keylist[0: 10]
  # printing result
  print(msg)
  for key in out:
    print(f"{key} : {d[key]}")

def import_data(file):
    print("Importing data ...")
    # read data
    df = pd.read_json(file,lines=True)#,lines=True,orient='records')
    df = df[all_feat]
    df = df.fillna(0) # only vote column has NaN
    #df.head(5)

    # data transformation
    le = preprocessing.LabelEncoder()
    change_cols = ["reviewerID" , "asin"]
    df["reviewerID"] = le.fit_transform(df["reviewerID"])
    max_rid = max(df["reviewerID"])
    df["asin"] = le.fit_transform(df["asin"])
    df["asin"] += max_rid
    df["verified"] = le.fit_transform(df["verified"])
    return df

def normalize(df):

    # normalize user rating to [-1,1], original scale [0,5]
    df["overall"] = (df["overall"] -2.5 ) / 2.5

    # normalize votes to [0,1]
    df["vote"] = df["vote"].replace(',','',regex=True)
    df["vote"] = df["vote"].astype(int)
    df["vote"] = df["vote"] /df["vote"].abs().max()

    # change verified to -1 vs. 1
    df["verified"] = df["verified"].replace(0,-1)

    print("#--------------- Normalized Data ---------------#")

    print(f'user rating normalized scale: [{df["overall"].min()}, {df["overall"].max()}]')
    print(f'vote normalized scale: [{df["vote"].min()}, {df["vote"].max()}]')
    print(f'verified value scale: [{df["verified"].min()}, {df["verified"].max()}]')
    print()
    return df

def observe_dataset(df):
    unverified_voted = []
    verified_novote = []
    voted_reviews = []
    noreviews = []

    for index, row in df.iterrows():
        if row['verified'] == -1:
          if row['vote'] > 0:
            rowdict = {'user': row['reviewerID'],
                       'verified': row['verified'],
                       'asin':row['asin'],
                       'rating': row['overall'],
                       'vote': row['vote']}
            unverified_voted.append(rowdict)

        elif row['vote'] == 0:
          rowdict = {'user': row['reviewerID'],
                     'verified': row['verified'],
                     'asin':row['asin'],
                     'rating': row['overall'],
                     'vote': row['vote']}
          verified_novote.append(rowdict)

        else:
          if row['reviewText'] or row['summary']:
            rowdict = {'user': row['reviewerID'],
                       'asin':row['asin'],
                       'rating': row['overall'],
                       'vote': row['vote'],
                       'reviewText':row['reviewText'],
                       'summary':row['summary']}
            voted_reviews.append(rowdict)

        if (not row['reviewText']) and (not row['summary']):
          rowdict = {'user': row['reviewerID'],
                     'asin':row['asin'],
                     'rating': row['overall'],
                     'vote': row['vote'],
                     'verified': row['verified'],
                     'reviewText':row['reviewText'],
                     'summary':row['summary']}
          noreviews.append(rowdict)

    print("#--------------- Dataset Info ---------------#")
    print(f"Unverified reviews received vote: {len(unverified_voted)}")
    print(f"Verified reviews received received no vote {len(verified_novote)}")
    print(f"Verified reviews no vote ratio {len(verified_novote)/len(df['reviewerID'])}")
    print(f"Verified voted reviews: {len(voted_reviews)}")
    print(f"No reviews: {len(noreviews)}")
    print()

def df_to_dict(df):
    # pick out nodes and weights, stored in list and dictionary
    graph_dict = {}



    return graph_dict

def df_to_graph(df):

    edges = []

    for index, row in df.iterrows():
        n1 = row['reviewerID']
        n2 = row['asin']
        edges.append((n1,n2))

    print("Adding nodes to graph ...")
    # add nodes to graph
    u_nodes, p_nodes = zip(*edges) # key = edge = (usernode, productnode)
    nodes = list(set(u_nodes + p_nodes))
    print(f"Users[{len(u_nodes)}] , Products[{len(p_nodes)}], Nodes[{len(nodes)}]")

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    #nx.set_edge_attributes(G, graph_dict)

    return G

# all nodes, not used...
def plot_whole_graph(G):
  plt.figure(figsize=(20,15))
  pos=nx.drawing.nx_pydot.pydot_layout(G, prog='fdp')
  #nx.draw(G, pos, node_color="#748AF9") # base color lightblue
  nx.draw(G,pos)
  labels = nx.get_edge_attributes(G,'weight')
  nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
  plt.draw()


# save to amazon_network_1000n.png
def plot_small_df(df):
  print("Ploting 1000 nodes sample network ...")
  small_df=df.drop(df.index[1000:])

  # {(node1,node2):weight}
  small_nl = []
  edge_list = []

  for index, row in small_df.iterrows():
      n1 = row['reviewerID']
      n2 = row['asin']
      edge = (n1,n2)
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

  plt.figure(figsize=(20,20))

  pos=nx.drawing.nx_pydot.pydot_layout(SG, prog='fdp')
  nx.draw(SG,pos)
  nx.draw_networkx_labels(SG,pos,node_lb)

  #nx.draw_shell(SG)
  #nx.draw_networkx_labels(SG,node_lb)
  plt.draw()
  plt.savefig('outputs/amazon_network_1000n.png')

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
  plt.savefig('outputs/amazon_network_histogram.png')


#@title def plot_degree_dens(G): plot density of degree distribution
# https://seaborn.pydata.org/generated/seaborn.kdeplot.html
def plot_degree_dens(G):
  print("Plot degree density graph ...")
  plt.figure(figsize=(10, 10))
  x = list(G.nodes)
  y = []
  for node in x:
    y.append(G.degree(node))

  # Make density plot
  sns.kdeplot(y)
  plt.xlim([0,1500])
  plt.savefig('outputs/amazon_network_degree_density.png')


#@title def get_astp(G): record average shortest path
#@markdown output: Nodes num: 55223, shortest path mean: 3.6516
# https://networkx.org/documentation/networkx-1.10/reference/generated/networkx.algorithms.shortest_paths.generic.average_shortest_path_length.html
def get_astp(G):
  print("Getting averaged shortest path ...")
  output_path='outputs/amazon_network_aveSTP_cluster.txt'

  length = nx.average_shortest_path_length(G)
  ave_clus = nx.average_clustering(G)
  with open(output_path, encoding='utf-8', mode='w+') as f:
      f.write(f'Averate Clustering Coefficients: {ave_clus}\n')
      f.write(f'Nodes num: {len(G.nodes())}, average shortest path: {length}\n')


if __name__ == "__main__":
    # import dataset
    file = amazon_data

    start = time.time()
    df = import_data(file)
    df = normalize(df)
    read_t = time.time()
    print(f"\nRead data time : {read_t - start}\n")

    observe_dataset(df)


    G = df_to_graph(df)
    graph_t = time.time()
    print(f"\nAdd to graph time : {graph_t - read_t}\n")

    # plot a small network structure sneakpeak
    plot_small_df(df)

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

    # get average shortest path
    get_astp(G)
    end = time.time()
    print(f"\nTotal runtime : {end - start}\n")
