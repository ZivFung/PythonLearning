import tensorflow as tf
# import cnn
import numpy as np
# import cv2 
import math
import h5py
from PIL import Image

sess=tf.Session()    
saver = tf.train.import_meta_graph('./checkpoint_dir/MyModel.meta')
saver.restore(sess,tf.train.latest_checkpoint('./checkpoint_dir'))
all_vars = tf.trainable_variables()
print(all_vars)
W1 = sess.run('W1_1:0')
W2 = sess.run('W2_1:0')
FCW = sess.run('fully_connected/weights:0')
FCB = sess.run('fully_connected/biases:0')
FCW1 =sess.run('FCWeights2_1:0') 
FCB1 =sess.run('FCBias2_1:0')
# # print(sess.run('W1_1:0'))
# # print(sess.run('W2_1:0'))
# W1 = np.array(sess.run('W1:0'))
# W2 = np.array(sess.run('W1:0'))

parameters = {"W1" : W1,
              "W2" : W2,
              "FCW1" : FCW,
              "FCBias1" : FCB,
              "FCW2" : FCW1,
              "FCBias2" : FCB1}

XTrain = np.zeros((1,32,32))
XTrain[0,:,:] = np.array(Image.open('F:/Study/Cnn/erzhi/Bool32/nine' + str(150) + '.bmp'))
XTrain[XTrain > 100 ] = 1
XTrain = XTrain[:,:,:,np.newaxis]
File = h5py.File('./Dataset4_number.h5','w')
File.create_dataset('train_set_x',data = XTrain)
File.close()
dataset = h5py.File('F:/Project/PYProject/FPGA_CNN/Dataset4_number.h5', "r")
TrainSetXOrig = np.array(dataset["train_set_x"][:])
(m,RowsNum,ColumnNum,DepthNum) = TrainSetXOrig.shape
X = tf.placeholder(tf.float32,shape=(None,RowsNum,ColumnNum,DepthNum))
X = tf.cast(TrainSetXOrig,tf.float32)

def CnnTest(X,parameters):
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
    # fc_w1 = FCWeights
    # fc_b1 = FCBias
    # XFullConnectData = tf.nn.relu(tf.matmul(XPooled3Data, fc_w1) + fc_b1)
    # XFullConnectData = tf.nn.relu(tf.matmul(XPooled3Data, fc_w1) + fc_b1)
    XFullConnectData = tf.matmul(XPooled3Data, fc_w1) + fc_b1
    # XFullConnectData2 = tf.nn.relu(tf.matmul(XFullConnectData, fc_w2) + fc_b2)
    return XFullConnectData



FullConnectData = CnnTest(X,parameters)
with tf.Session() as sess1: 
    print(sess1.run(FullConnectData))
