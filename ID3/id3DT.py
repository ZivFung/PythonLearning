from math import log
import operator



def creatdataSet():
    # 创建数据集
    # @return：
    #         dataSet - 数据集
    #         labels - 标签

    dataSet = [['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
               ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'],
               ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
               ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'],
               ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
               ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '是'],
               ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '是'],
               ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '是'],
               ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '否'],
               ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '否'],
               ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '否'],
               ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '否'],
               ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '否'],
               ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '否'],
               ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '否'],
               ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '否'],
               ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '否']]

    labels = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']

    return dataSet, labels


def splitDataSet(dataSet, axis, value):
    # 根据标签划分数据集
    # @Parameters：
    #             dataSet - 数据集
    #             axis - 划分数据集的特征
    #             value - 需要返回的特征的值
    # @Return：
    #       retDataSet - 划分后的数据集

    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)

    return retDataSet


def calcShannonEnt(dataSet):
    # 计算香农熵
	# @Parameters：
    #           dataSet - 数据集
    # @return:
    #         shannonEnt - 香农熵

    numEntires = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    shannonEnt = 0.0

    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntires
        shannonEnt -= prob * log(prob, 2)

    return shannonEnt


def chooseBestFeatureToSplit(dataSet):
    # 选择最优特征
    # @Parameters:
    #           dataSet - 数据集
    # @return:
    #       bestFeature - 最优特征

    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0

        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)

        infoGain = baseEntropy - newEntropy

        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i

    return bestFeature


def majorityCnt(classList):
    # 统计classlist中出现最多元素
    # Parameters:
    #           classList - 类标签列表
    # Returns:
    #       sortedClassCount[0][0] - 出现此处最多的元素(类标签)

    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
            classCount[vote] += 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]


def createTree(dataSet, labels, featLabels):
    classList = [example[-1] for example in dataSet]

    if classList.count(classList[0]) == len(classList):
        return classList[0]

    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    
    bestFeat = chooseBestFeatureToSplit(dataSet)
    print(str(bestFeat))
    bestFeatLabel = labels[bestFeat]
    print(str(bestFeatLabel))
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    print(str(labels))
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)

    for value in uniqueVals:
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), labels, featLabels)

    return myTree

if __name__ == '__main__':

    dataSet, labels = creatdataSet()
    print(dataSet, labels)
    featLabels = []
    DecisionTree = createTree(dataSet, labels, featLabels)
    print(DecisionTree)

