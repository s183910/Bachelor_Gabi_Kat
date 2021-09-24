
from itertools import combinations
# from tensorflow.keras.utils import plot_model, Progbar
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers, activations, losses, Model, Input
from tensorflow.nn import leaky_relu
import pdb
# import mymodule

# # generate data
# nb_query = 20
# query = np.array([i+1 for i in range(nb_query) for x in range(int(np.ceil(np.abs(np.random.normal(0,scale=15))+2)))])
# doc_features = np.random.random((len(query), 10))
# # Vi mener, at doc_features=documenter=grades, gender, stratum etc.

# doc_scores = np.random.randint(5, size=len(query)).astype(np.float32)
# # Vi mener, at scores netop er de qvantitative værdier for hvert document for hver student.


# Loading the data
df = pd.read_csv("src/Algoritmer/df_sum_score_py.csv")

# Creating the data
# features
df_document_features = df.copy()
df_document_features = df_document_features[["GENDER_bin", "HI_GRADE_AVG"]]
df_document_features = df_document_features.values.tolist()
doc_features = np.array(df_document_features)
#scores
doc_scores = df["COL_GRADE_AVG"].values

query = df.index.values



# put data into pairs
xi = []
xj = []
pij = []
pair_id = []
pair_query_id = []
pdb.set_trace()
for q in np.unique(query):
    query_idx = np.where(query == q)[0] # den når hertil og har empty query_idx
    pdb.set_trace()
    for pair_idx in combinations(query_idx, 2):
        pair_query_id.append(q)
        pair_id.append(pair_idx)
        i = pair_idx[0]
        j = pair_idx[1]
        xi.append(doc_features[i])
        xj.append(doc_features[j])

        if doc_scores[i] == doc_scores[j]:
            _pij = 0.5
        elif doc_scores[i] > doc_scores[j]:
            _pij = 1
        else:
            _pij = 0
        pij.append(_pij)

# pdb.set_trace()

xi = np.array(xi)
xj = np.array(xj)
pij = np.array(pij)
pair_query_id = np.array(pair_query_id)


xi_train, xi_test, xj_train, xj_test, pij_train, pij_test, pair_id_train, pair_id_test = train_test_split(
    xi, xj, pij, pair_id, test_size=0.2, stratify=pair_query_id)
