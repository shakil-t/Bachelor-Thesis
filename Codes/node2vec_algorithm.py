# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 12:54:49 2020

@author: shakil
"""

import networkx as nx
from node2vec import Node2Vec

def load_data(program_file_name):
    file=open(program_file_name, "r")
    read_to_scan=file.read()
    instances=read_to_scan.split("\n")
    data_set={i: [] for i in range(number_of_users+number_of_items)}
    for i in range(0, len(instances)-1):
        temp=instances[i].split()
        key=int(temp.pop(0))
        for item in temp:
            data_set[key].append(int(item))
    return data_set

number_of_users=943
number_of_items=1682

train_file="train.adjlist"

train_matrix=load_data(train_file)

train_subgraph=nx.from_dict_of_lists(train_matrix)
#print(nx.info(train_subgraph, n=1))

train_node2vec=Node2Vec(train_subgraph, dimensions=10, walk_length=5, num_walks=200, workers=1)
train_model=train_node2vec.fit(window=10, min_count=1, batch_words=4)
train_model.wv.most_similar('2')
train_model.wv.save_word2vec_format("train embedding.emb")
