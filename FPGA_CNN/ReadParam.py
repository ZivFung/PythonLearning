import tensorflow as tf
import numpy as np

sess=tf.Session()    
saver = tf.train.import_meta_graph('./checkpoint_dir/MyModel.meta')
saver.restore(sess,tf.train.latest_checkpoint('./checkpoint_dir'))
all_vars = tf.trainable_variables()
print(all_vars)

# FullConnect = np.array(sess.run('FCWeights_1:0'))
# W1 = np.array(sess.run('W1_1:0'))
# W2 = np.array(sess.run('W2_1:0'))
# np.savetxt("W1.txt",W1)
# np.savetxt("W2.txt",W2)
# np.savetxt("FCData.txt",FullConnect)
print(sess.run('W1_1:0'))
print(sess.run('W2_1:0'))
print(sess.run('FCWeights_1:0'))
print(sess.run('FCBias_1:0'))