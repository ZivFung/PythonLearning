# import numpy as np
# _*_coding:utf-8_*_
import math as ma
import operator
import uuid
import collections
import pydotplus
# from IPython.display import Image
from PIL import Image
import cairosvg as cairo
# import os
# import os
# import subprocess
# import platform
# import re
# from utils import log_utils


def calShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shanonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shanonEnt -= prob * ma.log(prob, 2)
    return shanonEnt


def createDataSet():
    # dataSet = [[1, 1, 'yes'],
    #            [1, 1, 'yes'],
    #            [1, 0, 'no'],
    #            [0, 1, 'no'],
    #            [0, 0, 'no']]
    dataSet = [['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', "是"],
               ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', "是"],
               ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', "是"],
               ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', "是"],
               ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', "是"],
               ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', "是"],
               ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', "是"],
               ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', "是"],

               ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', "否"],
               ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', "否"],
               ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', "否"],
               ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', "否"],
               ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', "否"],
               ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', "否"],
               ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', "否"],
               ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', "否"],
               ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', "否"]]
    # labels = ['no surfacing', 'flippers']
    # dataSet = np.mat(dataSet)
    # labels = np.mat(labels)
    labels = ["色泽", "根蒂", "敲声", "纹理", "脐部", "触感"]
    return dataSet, labels


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:      # abstract the fature
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1

    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueValue = set(featList)
        newEntropy = 0.0
        for value in uniqueValue:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
            classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def creatTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if (len(dataSet[0]) == 1):
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    print(str(bestFeatLabel))
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])

    featValues = [example[bestFeat] for example in dataSet]
    uniqueValues = set(featValues)
    for value in uniqueValues:
        subLabels = labels[:]
        print(str(labels))
        myTree[bestFeatLabel][value] = creatTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def get_nodes_edges(tree=None, root_node=None):
    ''' 返回树中所有节点和边
    '''
    Node = collections.namedtuple('Node', ['id', 'label'])
    Edge = collections.namedtuple('Edge', ['start', 'end', 'label'])
    # if tree is None:
    #     tree = self.tree
    if type(tree) is not dict:
        return [], []
    nodes, edges = [], []
    if root_node is None:
        label = list(tree.keys())[0]
        root_node = Node._make([uuid.uuid4(), label])
        nodes.append(root_node)
    for edge_label, sub_tree in tree[root_node.label].items():
        node_label = list(sub_tree.keys())[0] if type(sub_tree) is dict else sub_tree
        sub_node = Node._make([uuid.uuid4(), node_label])
        nodes.append(sub_node)
        edge = Edge._make([root_node, sub_node, edge_label])
        edges.append(edge)
        sub_nodes, sub_edges = get_nodes_edges(sub_tree, root_node=sub_node)
        nodes.extend(sub_nodes)
        edges.extend(sub_edges)
    return nodes, edges


def dotify(tree=None):
    ''' 获取树的Graphviz Dot文件的内容
    '''
    # if tree is None:
    #     tree = self.tree

    content = 'digraph decision_tree {\n'
    nodes, edges = get_nodes_edges(tree)
    for node in nodes:
        content += '    "{}" [label="{}"];\n'.format(node.id, node.label)
    for edge in edges:
        start, label, end = edge.start, edge.label, edge.end
        content += '    "{}" -> "{}" [label="{}"];\n'.format(start.id, end.id, label)
    content += '}'
    content.encode('utf-8')
    return content


def test():
    dataSet, labels = createDataSet()
    # chooseBestFeatureToSplit(dataSet)
    myTree = creatTree(dataSet, labels)
    with open('lenses.dot', 'w', encoding="utf-8") as f:
        dot = dotify(myTree)
        f.write(dot)
        graph = pydotplus.graph_from_dot_data(dot)
        graph.write_svg("WineTree.svg")
        cairo.svg2png("WineTree.svg")
        img = Image.open("WineTree.png")
        img.show()
    print(myTree)


def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


test()


# class Treenode(object):

#     def __init__(self, name, featureIndex, data):
#         self.data = data
#         self.name = name
#         self.featureIndex = featureIndex
#         # self.Rchild = None
#         # self.Lchild = None
#         self.child = []
#         self.childNum = 0
#         self.parent = None
#         self.parentChildIndex = 0
    
#     # def addRchild(self, node):
#     #     if self.Rchild is not None:
#     #         self.Rchild = node
#     #     else:
#     #         self.Rchild = node
#     #         self.childNum += 1
#     #     node.setParent(self)

#     # def addLchild(self, node):
#     #     if self.Lchild is not None:
#     #         self.Lchild = node
#     #     else:
#     #         self.Lchild = node
#     #         self.childNum += 1
#     #     node.setParent(self)

#     def addChild(self, node, data):
#         self.child[self.childNum] = node
#         self.childNum += 1
#         node.setParent(self)
#         node.parentChildIndex = self.childNum
#         self.data.append(data)

#     def deleteChild(self, childIndex):
#         self.child[childIndex - 1].parentChildIndex = 0
#         self.child[childIndex - 1].parent = None
#         self.child[childIndex - 1:] = self.child[childIndex:]
#         self.childNum -= 1

#     # def DeleteRchild(self):
#     #     self.Rchild = None  
#     #     self.childNum -= 1 

#     # def DeleteLchild(self):
#     #     self.Lchild = None  
#     #     self.childNum -= 1    
    
#     def setParent(self, node):  
#         self.parent = node 

#     def decisionProcess(dataSet):
#         if childNum != 0:
#             for i in range(len(self.data)):
#                 if dataSet[self.featureIndex] == data[i]:
#                     return self.child[i].decisionProcess(dataSet)
#             print("Classifer failure(No found of corresponding feature value)")
#             return None
#         # else:


# def calInformEnt(dataSet, featureIndex):
#     smpleNum = dataSet.shape[0]

#     labelCounts = {}

#     for i in range(smpleNum):
#         currentLabel = dataSet[i, featureIndex]
#         if currentLabel not in labelCounts.keys():
#             labelCounts[currentLabel] = 0
#         labelCounts[currentLabel] += 1
    
#     InformEnt = 0.0

#     for key in labelCounts:
#         prob = float(labelCounts[key]) / smpleNum
#         InformEnt -= prob * np.log(prob, 2)
#     return InformEnt


# def ID3DecisionTreeClassifier(dataSet):
#     sampleNum = dataSet.shape[0]
#     featureNum = dataSet.shape[1] - 1
#     featureValue = np.mat(np.zeros((featureNum, 2)))  # [FeatureIndex: FeatureType]
#     resultType = []  # [Resulttype: num]
#     resultNum = 0
#     # check the number of classify result
#     for i in range(sampleNum):
#         if dataSet[i: featureNum+1] not in resultType:
#             resultType[:0].append(dataSet[i: featureNum+1])
#             resultNum += 1
#             resultType[resultNum-1:1] += 1
#         else:
#             resultType[resultType[:0] == dataSet[i: featureNum+1]: 1] += 1

#     # check all the feature value and classify the feature
#     for i in range(featureNum):
#         for j in range(sampleNum):
#             if dataSet[i][j] not in featureValue[:, 0]:
#                 featureValue[i, :] = dataSet[i][j], len(dataSet[:i] == dataSet[i][j])

#     # calculate root infoemation entropy
#     maxRootEnt = 0
#     maxIndex = 0
#     for i in range(featureNum):
#         rootEnt[i] = calInformEnt(dataSet, featureIndex)
#         # if rootEnt[i] > maxRootEnt:
#         #     maxIndex = i
#         #     maxRootEnt = rootEnt[i]
    

#     rootNode = Treenode('root', 0)
