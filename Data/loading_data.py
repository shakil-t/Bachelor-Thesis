# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import random
from sklearn.model_selection import train_test_split

def load_data(program_file_name):
    file=open(program_file_name, "r")
    read_to_scan=file.read()
    instances=read_to_scan.split("\n")
    data_set=[]
    for i in range(0, len(instances)):
        temp=instances[i].split()
        #mapping the item vertices
        temp[1]=int(temp[1])+number_of_users
        data_set.append(temp[0:3])
    data_set=random.sample(data_set, len(data_set))
    data_set=np.asarray(data_set, dtype=int)
    return data_set

def building_rate_matrix(data_set):
    #np.zeros((number_of_users, number_of_items)) in case of need
    #rate_matrix=[[] for i in range(943)]
    rate_matrix={i: [] for i in range(number_of_users+number_of_items)}
    for i in range(0, len(data_set)):
        if data_set[i][2]>=rate:
            rate_matrix[data_set[i][0]-1].append(data_set[i][1]-1)
            rate_matrix[data_set[i][1]-1].append(data_set[i][0]-1)
    return rate_matrix

def split_data(rate_matrix):
    train_matrix={i: [] for i in range(number_of_users+number_of_items)}
    test_matrix={i: [] for i in range(number_of_users)}
    for i in range(0, number_of_users):
        if len(rate_matrix[i])>=10:
            train, test=train_test_split(rate_matrix[i], test_size=0.2)
            train_matrix[i]=train
            test_matrix[i]=test
    for i in range(0, number_of_users):
        for j in train_matrix[i]:
                train_matrix[j].append(i)
    return train_matrix, test_matrix

program_file_name="u.data"
number_of_users=943
number_of_items=1682
rate=4

data_set=load_data(program_file_name)

rate_matrix=building_rate_matrix(data_set)
train_matrix, test_matrix=split_data(rate_matrix)
with open("train.adjlist", "w") as ftrain:
    for i in range(0, number_of_users+number_of_items):
        ftrain.write(str(i)+" ")
        for item in train_matrix[i]:
            ftrain.write(str(item)+" ")
        ftrain.write("\n")
ftrain.close()
with open("test.txt", "w") as ftest:
    for i in range(0, number_of_users):
        ftest.write(str(i)+" ")
        for item in test_matrix[i]:
            ftest.write(str(item-number_of_users)+" ")
        ftest.write("\n")
ftest.close()

