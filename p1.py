#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 00:32:50 2018

@author: cristiano
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

dados = pd.read_csv('heart.csv', header = 1)

# colunas 'pressão arterial' e 'colesterol'
x = dados.iloc[:,[3,4]].values
scaler = StandardScaler()
x = scaler.fit_transform(x)

'''
# Para definir o número de clusters
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.xlabel('Número de clusters')
plt.ylabel('WCSS')
'''

kmeans = KMeans(n_clusters = 5, random_state = 0)
predicao = kmeans.fit_predict(x)

plt.figure(figsize=[12,8])
plt.scatter(x[predicao == 0, 0], x[predicao == 0, 1], s = 200, c = 'green', label = 'Cluster 1')
plt.scatter(x[predicao == 1, 0], x[predicao == 1, 1], s = 200, c = 'blue', label = 'Cluster 2')
plt.scatter(x[predicao == 2, 0], x[predicao == 2, 1], s = 200, c = 'red', label = 'Cluster 3')
plt.scatter(x[predicao == 3, 0], x[predicao == 3, 1], s = 200, c = 'pink', label = 'Cluster 4')
plt.scatter(x[predicao == 4, 0], x[predicao == 4, 1], s = 200, c = 'orange', label = 'Cluster 5')
#plt.scatter(x[predicao == 5, 0], x[predicao == 5, 1], s = 200, c = 'yellow', label = 'Cluster 6')
#plt.scatter(x[predicao == 6, 0], x[predicao == 6, 1], s = 200, c = 'gray', label = 'Cluster 7')
#plt.scatter(x[predicao == 7, 0], x[predicao == 7, 1], s = 200, c = 'black', label = 'Cluster 8')
plt.xlabel('Pressão Arterial')
plt.ylabel('Colesterol')
plt.legend()


#lista_pacientes = np.column_stack((dados, predicao))
#lista_pacientes = lista_pacientes[lista_pacientes[:,14].argsort]




