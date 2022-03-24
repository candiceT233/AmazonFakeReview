import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter
from tabulate import tabulate
import networkx as nx
import glob
#print(glob.glob("/home/adam/*"))


# global variables
data_path="/content/drive/MyDrive/CS579/CS579-Project2/datasets"
# https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html
otc_data="soc-sign-bitcoinotc.csv"
# https://snap.stanford.edu/data/soc-sign-epinions.html
ep_data="soc-sign-epinions.txt"

# http://snap.stanford.edu/data/amazon0601.html
amazon_meta="amazon-meta.txt"
amazon_data="amazon0601.txt"
