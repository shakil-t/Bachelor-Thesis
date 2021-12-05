# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 16:30:29 2020

@author: shakil
"""

import numpy as np
from sklearn.metrics import ndcg_score

def load_data(program_file_name):
    file=open(program_file_name, "r")
    read_to_scan=file.read()
    instances=read_to_scan.split("\n")
    data_set={i: [] for i in range(number_of_users)}
    for i in range(0, number_of_users):
        temp=instances[i].split()
        key=int(temp.pop(0))
        for item in temp:
            data_set[key].append(int(item))
    return data_set

def load_vectors(program_file_name):
    file=open(program_file_name, "r")
    read_to_scan=file.read()
    instances=read_to_scan.split("\n")
    data_set={i: [] for i in range(number_of_users+number_of_items)}
    for i in range(1, len(instances)-1):
        temp=instances[i].split()
        key=int(temp.pop(0))
        temp=[float(i) for i in temp]
        data_set[key]=list(temp)
    return data_set

def calculate_inner_product(adj, vec, n):
    inner_product={i:[] for i in range(number_of_users)}
    temp2=[]
    for i in range(0, number_of_users):
        temp1=[]
        for j in range(number_of_users, number_of_users+number_of_items):
            if j not in adj[i]:
                dot_product=np.inner(np.array(vec[i]), np.array(vec[j]))
                temp1.append((i, j, dot_product))
        temp1.sort(key=lambda x:x[2], reverse=True)
        temp2.append(temp1[0:n])
    for i in range(0, number_of_users):
        temp1=[]
        for item in temp2[i]:
            temp1.append(item[1]-number_of_users)
        inner_product[i]=temp1
    return inner_product
    
def calculate_cosine_similarity(adj, vec, n):
    cosine_similarity={i:[] for i in range(number_of_users)}
    temp2=[]
    for i in range(0, number_of_users):
        temp1=[]
        for j in range(number_of_users, number_of_users+number_of_items):
            if j not in adj[i]:
                dot_product=np.inner(np.array(vec[i]), np.array(vec[j]))
                v1=np.linalg.norm(np.array(vec[i]))
                v2=np.linalg.norm(np.array(vec[j]))
                similarity=(dot_product)/(v1*v2)
                temp1.append((i, j, similarity))
        temp1.sort(key=lambda x:x[2], reverse=True)
        temp2.append(temp1[0:n])
    for i in range(0, number_of_users):
        temp1=[]
        for item in temp2[i]:
            temp1.append(item[1]-number_of_users)
        cosine_similarity[i]=temp1
    return cosine_similarity

#r stands for recommended items for users
def calculate_precision(r):
    precision=[0 for i in range(number_of_users)]
    for i in range(0, number_of_users):
        precision[i]=len(set(test_matrix[i]).intersection(r[i]))/n
    return precision, sum(precision)/number_of_users

def calculate_recall(r):
    recall=[0 for i in range(number_of_users)]
    for i in range(0, number_of_users):
        if len(test_matrix[i])!=0:
            recall[i]=len(set(test_matrix[i]).intersection(r[i]))/len(test_matrix[i])
        else:
            recall[i]=0
    return recall, sum(recall)/number_of_users

def calculate_f1(precision, recall):
    f1=[0 for i in range(number_of_users)]
    for i in range(0, number_of_users):
        if precision[i]==0 and recall[i]==0:
            f1[i]=0
        else:
            f1[i]=2*((precision[i]*recall[i])/(precision[i]+recall[i]))
    return sum(f1)/number_of_users

def calculate_score(a, b):
    score=[0 for i in range(n)]
    for i in range(0, n):
        if a[i] in b:
            score[i]=1
    return score

def calculate_ndcg(r):
    ndcg=[0 for i in range(number_of_users)]
    for i in range(0, number_of_users):
        if len(test_matrix[i])>=n:
            score=calculate_score(test_matrix[i][0:n], r[i])
            ndcg[i]=ndcg_score(np.array([test_matrix[i][0:n]]), np.array([score]))
        else:
            ndcg[i]=0
    return sum(ndcg)/number_of_users

def calculate_1call(r):
    call=[0 for i in range(number_of_users)]
    for i in range(0, number_of_users):
        if len(set(test_matrix[i]).intersection(r[i]))>0:
            call[i]=1
        else:
            call[i]=0
    return sum(call)/number_of_users
    
number_of_users=943
number_of_items=1682
n=20

#test and train data
train_file="train.adjlist"
test_file="test.txt"
train_matrix=load_data(train_file)
test_matrix=load_data(test_file)

#loading vectors in both algorithms
dw_train_file="C:/Users/shakil/train.embeddings"
n2v_train_file="train embedding.emb"
dw_train_matrix=load_vectors(dw_train_file)
n2v_train_matrix=load_vectors(n2v_train_file)

#calculating inner product for both algorithmes
dw_train_inner_matrix=calculate_inner_product(train_matrix, dw_train_matrix, n)
n2v_train_inner_matrix=calculate_inner_product(train_matrix, n2v_train_matrix, n)

#calculating cosine similarity for both algorithmes
dw_train_cosine_matrix=calculate_cosine_similarity(train_matrix, dw_train_matrix, n)
n2v_train_cosine_matrix=calculate_cosine_similarity(train_matrix, n2v_train_matrix, n)

#evaluating metrics
print("Deepwalk Algorithm")
print("Evaluating metrics by calculating inner products:")
user_precision1, precision1=calculate_precision(dw_train_inner_matrix)
user_recall1, recall1=calculate_recall(dw_train_inner_matrix)
f11=calculate_f1(user_precision1, user_recall1)
ndcg1=calculate_ndcg(dw_train_inner_matrix)
call1=calculate_1call(dw_train_inner_matrix)
print("percision=", precision1)
print("Recall=", recall1)
print("F1=", f11)
print("NDCG=", ndcg1)
print("1Call=", call1)
print("Evaluating metrics by calculating cosine similarity:")
user_precision2, precision2=calculate_precision(dw_train_cosine_matrix)
user_recall2, recall2=calculate_recall(dw_train_cosine_matrix)
f12=calculate_f1(user_precision2, user_recall2)
ndcg2=calculate_ndcg(dw_train_cosine_matrix)
call2=calculate_1call(dw_train_cosine_matrix)
print("percision=", precision2)
print("Recall=", recall2)
print("F1=", f12)
print("NDCG=", ndcg2)
print("1Call=", call2)

print()

print("Node2Vec Algorithm")
print("Evaluating metrics by calculating inner products:")
user_precision3, precision3=calculate_precision(n2v_train_inner_matrix)
user_recall3, recall3=calculate_recall(n2v_train_inner_matrix)
f13=calculate_f1(user_precision3, user_recall3)
ndcg3=calculate_ndcg(n2v_train_inner_matrix)
call3=calculate_1call(n2v_train_inner_matrix)
print("percision=", precision3)
print("Recall=", recall3)
print("F1=", f13)
print("NDCG=", ndcg3)
print("1Call=", call3)
print("Evaluating metrics by calculating cosine similarity:")
user_precision4, precision4=calculate_precision(n2v_train_cosine_matrix)
user_recall4, recall4=calculate_recall(n2v_train_cosine_matrix)
f14=calculate_f1(user_precision4, user_recall4)
ndcg4=calculate_ndcg(n2v_train_cosine_matrix)
call4=calculate_1call(n2v_train_cosine_matrix)
print("percision=", precision4)
print("Recall=", recall4)
print("F1=", f14)
print("NDCG=", ndcg4)
print("1Call=", call4)