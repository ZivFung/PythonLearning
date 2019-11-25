import numpy as np
from k_means import Kmeans   
from k_means import ShowCluster  

DataSet = []
Filein = open('./testSet.txt')
for lines in Filein.readlines():
    lineArr = lines.strip().split('\t')
    DataSet.append([float(lineArr[0]), float(lineArr[1])])

DataSet = np.mat(DataSet)
print("Dataset shape:" + str(DataSet.shape))

k = 4
Centroids, ClusterAssment = Kmeans(DataSet, k)
ShowCluster(DataSet, k, Centroids, ClusterAssment)
