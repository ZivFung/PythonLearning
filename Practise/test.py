import tensorflow as tf  
  
a=tf.constant(  
        [[1.0,2.0,3.0,4.0],  
        [5.0,6.0,7.0,8.0],  
        [8.0,7.0,6.0,5.0],  
        [4.0,3.0,2.0,1.0]]
    )  
  
a=tf.reshape(a,[1,4,4,1])  

filter=tf.constant([  
        [[1.0],  
        # [0,1.0]],  

        [2.0],
        [3.0]]  
        #  [1.0,0]]  
    ])  
  
filter=tf.reshape(filter,[1,1,1,3])  

filter2=tf.constant([  
        [[1.0,0.0],  
        [0,0.0]],  

        [[0.0,0.0], 
        [1.0,0]],

        [[0.0,0.0], 
        [1.0,0]],

        [[0.0,0.0], 
        [1.0,0]],

        [[0.0,0.0], 
        [1.0,0]],

        [[0.0,1.0], 
        [0.0,0]],

        [[1.0,0.0],  
        [0,1.0]],  

        [[1.0,0.0], 
        [1.0,0]],

        [[0.0,1.0], 
        [1.0,0]]
    ])  
  
filter2=tf.reshape(filter2,[2,2,3,3])  
'''input = tf.Variable(tf.random_normal([1,3,3,5]))  
filter = tf.Variable(tf.random_normal([1,1,5,1]))  
'''  
filter3=tf.constant([  
    [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8,9],
    [2,4,6,8,10,12,14,16,18],
    [1,3,5,7,9,11,13,15,17],

    [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8,9],
    [1,3,5,7,9,11,13,15,17],
    [2,4,6,8,10,12,14,16,18],

    [2,4,6,8,10,12,14,16,18],
    [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8,9],
    [1,3,5,7,9,11,13,15,17]
    ])  
filter3=tf.reshape(filter3,[27,3]) 
op = tf.nn.conv2d(a, filter, strides=[1, 1, 1, 1], padding='VALID')  
relu = tf.nn.relu(op,name='Relu1_1')
pool = tf.nn.max_pool(relu,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'VALID',name='Pooling1_1')

cov2 = tf.nn.conv2d(pool, filter2, strides=[1, 1, 1, 1], padding='VALID')  
XPooled3Data = tf.contrib.layers.flatten(cov2)
# fc_w1 = FCWeights
# fc_b1 = FCBias
# XFullConnectData = tf.matmul(XPooled3Data, filter3)
with tf.Session() as sess:  
    print("image:")  
    image=sess.run(a)  
    print (image)
    print("filter1:")
    kernel=sess.run(filter)  
    print(kernel)  
    print("reslut:")  
    result=sess.run(pool)  
    print (result) 
    # print("relu:")  
    # reluresult=sess.run(relu)  
    # print (reluresult) 
    # print("pool:")  
    # poolresult=sess.run(pool)  
    # print (poolresult) 
    print("filter2:")
    kernel=sess.run(filter2)  
    print(kernel)  
    print("cov2:")  
    cov2result=sess.run(cov2)  
    print (cov2result)
    print("Flatten:")  
    flatresult=sess.run(XPooled3Data)  
    print (flatresult) 
    print("filter3:")
    kernel=sess.run(filter3)  
    print(kernel)   
    # print("FCResult:")
    # kernel=sess.run(XFullConnectData)  
    # print(kernel)  