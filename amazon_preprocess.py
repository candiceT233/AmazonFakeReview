"""
Notes:

edge weight := w1*Goodness + w2*Fairness

# node ettributes:
- user node: vote (normalized) from scale [0,1]
- product node: averaged overall rating from all users [-1, 1] (normalized)

# edge attribute definition:
- Goodness: TF-IDF value
- Fairness: difference of user rating and averaged product overall rating value:
-- (user_rating - product_rating) + 1
-- if product rating is 1, user rating is -1, F = -|-1-1| +1 = -1 (lowest)
-- if product rating is 1, user rating is 0, then F = -|0-1| +1= 0
-- if product rating is 0, user rating is 0, F = -0 + 1 = 1 (highest)

- Ground truth (label): -1 vs. 1
Unverified: -1
Verified: (0 + normalized vote + TF-IDF value)/3
negative values assign to -1
positive value assign to 1
Predit Fraud Reviews using TD-IDF, not per user base, assume 50% fake
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter
#from tabulate import tabulate
import networkx as nx # version networkx=2.3
import seaborn as sns
from collections import Counter
import math
import string
import time

from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer


# ---- global variables
rate_feat = ["overall", "verified", "reviewerID", "asin", "vote"]
review_feat = ["reviewerID", "asin", "reviewText", "summary"]
all_feat = ["reviewerID", "asin", "overall", "verified", "vote", "reviewText", "summary"]

# Video Game product text reviews
# info site: https://nijianmo.github.io/amazon/index.html
# download site: http://deepyeti.ucsd.edu/jianmo/amazon/index.html
amazon_data="datasets/Video_Games_5.json"

# english stop words doc
# source: https://gist.github.com/sebleier/554280
stopwords_file="datasets/english_stopwords.txt"


def print_ten_sample(d, msg="print check: "):
  keylist = list(d.keys()) # edge list
  out = keylist[0: 10]
  # printing result
  print(msg)
  for key in out:
    print(f"{key} : {d[key]}")


def read_data(file):
    # read data
    df = pd.read_json(file,lines=True)#,lines=True,orient='records')

    """
    with open(file, encoding='utf-8-sig') as fp:
        data = json.loads(''.join(line.strip() for line in fp))
    df = pd.json_normalize(data)
    """

    df = df[all_feat]
    df = df.fillna(0) # only vote column has NaN

    # data transformation
    le = preprocessing.LabelEncoder()
    change_cols = ["reviewerID" , "asin"]

    #df["reviewID", "asin"] = le.fit_transform(df["reviewID", "asin"])
    df["reviewerID"] = le.fit_transform(df["reviewerID"])
    max_rid = max(df["reviewerID"])
    df["asin"] = le.fit_transform(df["asin"])
    df["asin"] += max_rid
    df["verified"] = le.fit_transform(df["verified"])
    #df[cols] = df[cols].apply(le.fit_transform)
    print("Show data sample: ")
    print(df.head(5))
    print(df.shape)
    return df

def normalize(df):
    # normalize user rating to [-1,1], original scale [0,5]
    df["overall"] = (df["overall"] -2.5 ) / 2.5


    # normalize votes to [0,1]
    df["vote"] = df["vote"].replace(',','',regex=True)
    df["vote"] = df["vote"].astype(int)
    df["vote"] = df["vote"] /df["vote"].abs().max()

    # change verified to -1 vs. 0
    df["verified"] = df["verified"].replace(0,-1)
    df["verified"] = df["verified"].replace(1,0)

    print("#--------------- Normalized Data ---------------#")
    print(f'user rating normalized scale: [{df["overall"].min()}, {df["overall"].max()}]')
    print(f'vote normalized scale: [{df["vote"].min()}, {df["vote"].max()}]')
    print(f'verified value scale: [{df["verified"].min()}, {df["verified"].max()}]')
    print()


    return df


def text_proc(df):
    print("Preprocessing review texts...")
    # get stopwords
    file1 = open(stopwords_file, 'r')
    stopwords = list(file1.readlines())
    stopwords = [str(word).replace('\n','') for word in stopwords]

    # preprocessing 1: split strings into word list
    fulldoc_dict = {}
    for index, row in df.iterrows():
        edge = f"{row['reviewerID']},{row['asin']}"
        if edge not in fulldoc_dict.keys():
            fulldoc_dict[edge] = []
        for word in list(str(row['reviewText']).replace(',', ' ').split()):
            fulldoc_dict[edge].append(word)
        for word in list(str(row['summary']).replace(',', ' ').split()):
            fulldoc_dict[edge].append(word)

    # preprocessing 2: remove common punctuations
    for key in fulldoc_dict.keys():
        fulldoc_dict[key] = [''.join(c for c in s if c not in string.punctuation) for s in fulldoc_dict[key]]

    # preprocessing 3: remove common stopwords
    for key in fulldoc_dict.keys():
        w_list = list(fulldoc_dict[key])
        for sw in stopwords:
            while sw in w_list:
                w_list.remove(sw)
        fulldoc_dict[key] = w_list

    return fulldoc_dict

def get_tfidf_score_dict(df, fulldoc_dict):
    print("Getting review tfidf scores...")
    # dictionary to dataframe
    full_doc = []
    for key in fulldoc_dict.keys():
        for word in fulldoc_dict[key]:
            full_doc.append(word)

    # tf-idf scoring
    vectorizer = TfidfVectorizer(use_idf=True)
    vec = vectorizer.fit_transform(full_doc)
    df_tfidf = pd.DataFrame(vec[0].T.todense(), index=vectorizer.get_feature_names_out(), columns=["TF-IDF"])
    df_tfidf = df_tfidf.sort_values('TF-IDF', ascending=False)

    # get tfidf values into dictionary
    tfidf_dict = {}
    for index, row in df_tfidf.iterrows():
        tfidf_dict[index] = row['TF-IDF']

    score_dict = {}

    for key in fulldoc_dict.keys():
        score = 0
        for word in list(fulldoc_dict[key]):
            if word in tfidf_dict:
                score += tfidf_dict[word]
        score_dict[key] = score

    return score_dict

def tfidf_score_todf(df, score_dict):
    print("Adding tfidf scores to df...")
    # normalize scores
    score_max = max(score_dict.values())
    for key in score_dict.keys():
        score_dict[key] = score_dict[key]/score_max

    #print_ten_sample(score_dict)
    score_list = []
    for index, row in df.iterrows():
        edge = f"{row['reviewerID']},{row['asin']}"
        score_list.append(score_dict[edge])

    df['tfidf'] = score_list

def get_tfidf(df_all):
    # review_feat = ["reviewerID", "asin", "reviewText", "summary"]
    df = df_all[review_feat]
    df["reviewText"] = df["reviewText"].str.lower()
    df["summary"] = df["summary"].str.lower()

    fulldoc_dict = text_proc(df)
    score_dict = get_tfidf_score_dict(df, fulldoc_dict)
    tfidf_score_todf(df_all, score_dict)
    return df_all

def output_for_ml(df,filename="datasets/amazon_ml_1.csv"):
    print(f"Outputting dataframe to {filename} ...")
    #print(df.columns)
    df_feat = ['reviewerID', 'asin', 'overall', 'verified', 'vote', 'reviewText',
       'summary', 'tfidf', 'fairness', 'is_trust']
    out_feat = ['EdgeID','goodness','fairness','is_trust']
    # TODO
    df = df.rename(columns={'tfidf':'goodness'})

    # ['reviewerID','asin','vote','fairness','is_strust']
    # combine reviewerID and asin into single edge ID
    id_list = []
    for index, row in df.iterrows():
        edgeid = str(row['reviewerID']) + " " + str(row['asin'])
        id_list.append(edgeid)
    df['EdgeID'] = id_list
    df = df[out_feat]
    df.to_csv(filename)

def get_target(df):

    trust_list = []
    for index, row in df.iterrows():
        val = 0
        if row['verified'] < 0:
            val += -1
        val += row['tfidf']
        val += row['vote']
        val = val/3
        trust_list.append(val)

    # Ground Truth to be ~ 50% negative and positive
    percent = 50
    values = list(sorted(np.asarray(trust_list, dtype=float))[:int(len(trust_list)/(100/percent))])
    threshold = max(values)

    trust_val = []
    for val in trust_list:
        if val < threshold:
            trust_val.append(-1)
        else:
            trust_val.append(1)


    df['is_trust'] = trust_val
    return df


def get_graph_dict(df):
    print("Storing nodes and edges into dictionary ...")

    graph_dict = {}

    for index, row in df.iterrows():
        n1 = row['reviewerID']
        n2 = row['asin']
        rate = row['overall']
        key = (n1,n2)
        graph_dict[key] = {"rate": rate, "vote": row["vote"], "verified": row["verified"]}

    return graph_dict

def create_graph(graph_dict):
    print("Creating graph ...")
    u_nodes, p_nodes = zip(*graph_dict.keys()) # key = edge = (usernode, productnode)
    nodes = list(set(u_nodes + p_nodes))

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(list(graph_dict.keys()))

    nx.set_edge_attributes(G, graph_dict)
    return G

def get_ave_rating(df,graph_dict):
    print("Getting averaged product rating ...")
    """
    product node:
    - rating: averaged overall rating from all users [-1, 1]
    """
    p_nodes = list(df["asin"])
    G = create_graph(graph_dict)
    max_indeg = 0

    for node in p_nodes:
        edges = G.in_edges(node)
        if max_indeg < len(edges):
            max_indeg = len(edges)

    p_dict = {}
    for node in p_nodes:
        edges = G.in_edges(node)
        #print(edges)
        sum_rating = 0
        for edge in edges:
            sum_rating += graph_dict[edge]['rate']
        p_dict[node] = {"sum_rating": sum_rating, "num_rate": len(edges) }

    # insert as average rating to dataset for corresponding product edge
    # remove the rating of current edge
    rate_list = []
    for index, row in df.iterrows():
        node = row['asin']
        num_rate = p_dict[node]['num_rate']
        if (num_rate - 1 ) == 0:
            ave_rate = p_dict[node]['sum_rating'] - row['overall']
        else:
            ave_rate = (p_dict[node]['sum_rating'] - row['overall'])/(p_dict[node]['num_rate'] -1)
        rate_list.append(ave_rate)

    df['ave_rate'] = rate_list

    return df


def get_fairness(df):
    """
    Fairness: difference of user rating and averaged product overall rating value:
    - |user_rating - product_rating| + 1
    """

    graph_dict = get_graph_dict(df)
    df = get_ave_rating(df,graph_dict)

    # calculate fairness
    print("Calculating fairness ...")
    fair_list = []
    for index, row in df.iterrows():
        u_rate = row['overall']
        p_rate = row['ave_rate']
        fair = -1*abs(u_rate - p_rate) + 1
        fair_list.append(fair)

    df['fairness'] = fair_list

    return df

if __name__ == "__main__":

    start = time.time()
    df = read_data(amazon_data)
    df = normalize(df)
    read_t = time.time()
    print(f"\nRead data time : {read_t - start}\n")

    # get fairness feature
    df = get_tfidf(df) #working
    df = get_fairness(df) #working

    proc_t = time.time()
    print(f"\nFeature extraction time : {proc_t - read_t}\n")

    # dummy values
    #df['tfidf'] = np.random.rand(df.shape[0])
    #df['fairness'] = np.random.rand(df.shape[0])

    df = get_target(df) # working
    label_t = time.time()
    print(f"\nLabeling time : {label_t - proc_t}\n")


    output_for_ml(df)
    end = time.time()
    print(f"\nOutput time : {end - label_t}\n")
    print(f"\nOverall runtime : {end - start}\n")
