import tensorflow as tf
import h5py
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
from tensorflow.python.framework import ops
from PIL import Image
from scipy import ndimage

def LoadDataset():
    dataset = h5py.File('F:/Project/PYProject/FPGA_CNN/Dataset3_number.h5', "r")
    TrainSetXOrig = np.array(dataset["train_set_x"][:])
    TrainSetYOrig = np.array(dataset["train_set_y"][:])
    TestSetXOrig = np.array(dataset["test_set_x"][:])
    TestSetYOrig = np.array(dataset["test_set_y"][:])
    return TrainSetXOrig,TrainSetYOrig,TestSetXOrig,TestSetYOrig

XTrainOrig,YTrainOrig,XTestOrig,YTestOrig = LoadDataset()

XTrain = XTrainOrig
XTest = XTestOrig
YTrain = YTrainOrig
YTest = YTestOrig
print("number of training examples =" + str(XTrain.shape[0]))
print("number of test examples =" + str(XTest.shape[0]))
print("X_train shape:" + str(XTrain.shape))
print("Y_train shape:" + str(YTrain.shape))
print("X_test shape:" + str(XTest.shape))
print("Y_test shape:" + str(YTest.shape))
conv_layers = {}

'''
number of training examples = 29
number of test examples = 8
XTrain shape : (29,32,32,1)
YTrain shape : (29,4)
XTest shape : (8,32,32,1)
YTest shape : (8,4)
'''
def CreatePlaceholders(RowsNum,ColumnNum,DepthNum,YNum):
    with tf.name_scope('InputData'):
        X = tf.placeholder(tf.float32,shape=(None,RowsNum,ColumnNum,DepthNum))
        Y = tf.placeholder(tf.float32,shape=(None,YNum))
    return X,Y

def InitializeParameters():
    with tf.name_scope('W1'):
        W1 = tf.get_variable("W1",[5,5,1,3],initializer=tf.contrib.layers.xavier_initializer(seed = 0))
    with tf.name_scope('W2'):
        W2 = tf.get_variable("W2",[5,5,3,5],initializer= tf.contrib.layers.xavier_initializer(seed = 0)) 

    with tf.name_scope('FCWeights1'):
        FCWeights = tf.get_variable("FCWeights1",[125,10],initializer=tf.contrib.layers.xavier_initializer(seed = 0))
    with tf.name_scope('FCBias1'):
        FCBias = tf.get_variable("FCBias1",[10,],initializer= tf.contrib.layers.xavier_initializer(seed = 0)) 
    with tf.name_scope('FCWeights2'):
        FCWeights2 = tf.get_variable("FCWeights2",[50,10],initializer=tf.contrib.layers.xavier_initializer(seed = 0))
    with tf.name_scope('FCBias2'):
        FCBias2 = tf.get_variable("FCBias2",[10,],initializer= tf.contrib.layers.xavier_initializer(seed = 0)) 
    # tf.summary.histogram('./graphs/histogram',W1)
    parameters = {"W1" : W1,
                  "W2" : W2,
                  "FCW1" : FCWeights,
                  "FCBias1" : FCBias,
                  "FCW2" : FCWeights2,
                  "FCBias2" : FCBias2}
    return parameters

def ForwardPropagation(X,parameters):
    W1 = parameters['W1'] 
    W2 = parameters['W2']
    fc_w1 = parameters['FCW1']
    fc_b1 = parameters['FCBias1']
    fc_w2 = parameters["FCW2"]
    fc_b2 = parameters["FCBias2"]

    with tf.name_scope('layer1'):
        # XPooled1Data = tf.nn.max_pool(X,ksize = [1,15,20,1],strides = [1,15,20,1],padding = 'VALID',name='Pooling1')
        XConv1Data = tf.nn.conv2d(X,W1,strides = [1,1,1,1],padding = 'VALID',name='cov1_1')
        XActivate1Data = tf.nn.relu(XConv1Data,name='Relu1_1')
        XPooled2Data = tf.nn.max_pool(XActivate1Data,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'VALID',name='Pooling1_1')
    with tf.name_scope('layer2'):
        XConv2Data = tf.nn.conv2d(XPooled2Data,W2,strides = [1,1,1,1],padding = 'VALID')
        XActivate2Data = tf.nn.sigmoid(XConv2Data)
        # XActivate2Data = tf.nn.relu(XConv2Data,name='Relu2_1')
        XPooled3Data = tf.nn.max_pool(XActivate2Data,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'VALID',name='pooling2_1')
    XPooled3Data = tf.contrib.layers.flatten(XPooled3Data)
    XFullConnectData = tf.contrib.layers.fully_connected(XPooled3Data,num_outputs = 10,activation_fn = None,variables_collections='conv_layers')
    # XFullConnectData = tf.nn.relu(tf.matmul(XPooled3Data, fc_w1) + fc_b1)
    # XFullConnectData2 = tf.nn.relu(tf.matmul(XFullConnectData, fc_w2) + fc_b2)
    XSoftMaxData = tf.nn.softmax(XFullConnectData)
    return XSoftMaxData

def ComputeLoss(FullConnectData,Y):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = FullConnectData,labels = Y))
    return loss

def RandomMinibatches(X,Y,mini_batch_size = 64):
    m = X.shape[0]
    mini_batches = []
    
    #Step1: Shuffle(X,Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    #Step2: Partition(shuffled_X,shuffled_Y).Minus the end case'
    num_complete_minibatches = math.floor(m/mini_batch_size)#
    for k in range(0,num_complete_minibatches):
        mini_batch_X = shuffled_X[k*mini_batch_size : k*mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k*mini_batch_size : k*mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)

    if m%mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

def model(XTrain, YTrain, XTest, YTest, learning_rate,num_epochs = 100, minibatch_size = 6,print_cost = True):
    (m,RowsNum,ColumnNum,DepthNum) = XTrain.shape
    YNum = YTrain.shape[1]
    costs = []
    X,Y = CreatePlaceholders(RowsNum,ColumnNum,DepthNum,YNum)

    parameters = InitializeParameters()

    FullConnectData = ForwardPropagation(X,parameters)

    loss = ComputeLoss(FullConnectData,Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        
        sess.run(init)

        for epoch in range(num_epochs):
            minibatch_cost = 0
            num_minibatches = int(m/minibatch_size)
            minibatches = RandomMinibatches(XTrain,YTrain,minibatch_size)

            for minibatch in minibatches:

                (minibatch_X,minibatch_Y) = minibatch
                _,temp_loss = sess.run([optimizer,loss],feed_dict={X:minibatch_X,Y:minibatch_Y})
                
                minibatch_cost += temp_loss / num_minibatches

            # wirter.add_summary(summary, epoch)
            if print_cost == True and epoch % 5 == 0:
                print("Cost after epoch %i : %f"%(epoch,minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)

        

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        saver = tf.train.Saver()
        saver.save(sess,"./checkpoint_dir/MyModel")

        predict_op = tf.argmax(FullConnectData,1)

        correct_prediction = tf.equal(predict_op,tf.argmax(Y,1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X:XTrain, Y:YTrain})
        test_accuracy = accuracy.eval({X:XTest,Y:YTest})
        print("Train Accuarcy:",train_accuracy)
        print("Test Accuarcy:",test_accuracy)
        wirter = tf.summary.FileWriter('./graphs',sess.graph)
        wirter.close()

    return train_accuracy,test_accuracy,parameters

_,_,parameters=model(XTrain, YTrain, XTest, YTest, learning_rate = 0.0001,num_epochs = 300, minibatch_size = 5, print_cost = True)

        