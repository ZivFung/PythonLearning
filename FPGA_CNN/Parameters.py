import tensorflow as tf
import numpy as np
from xlwt import Workbook
book=Workbook()

sess=tf.Session()    
saver = tf.train.import_meta_graph('./checkpoint_dir/MyModel.meta')
saver.restore(sess,tf.train.latest_checkpoint('./checkpoint_dir'))
all_vars = tf.trainable_variables()
print(all_vars)

w1 = np.array(sess.run('W1_1:0'),np.str_)
w2 = np.array(sess.run('W2_1:0'),np.str_)
w3 = np.array(sess.run('fully_connected/weights:0'),np.str_)
b =np.array(sess.run('fully_connected/biases:0'),np.str_)

sheet1=book.add_sheet('w1_1')
sheet2=book.add_sheet('w1_2')
sheet3=book.add_sheet('w1_3')
sheet4=book.add_sheet('w2_1_1')
sheet5=book.add_sheet('w2_1_2')
sheet6=book.add_sheet('w2_1_3')
sheet7=book.add_sheet('w2_1_4')
sheet8=book.add_sheet('w2_1_5')
sheet9=book.add_sheet('w2_2_1')
sheet10=book.add_sheet('w2_2_2')
sheet11=book.add_sheet('w2_2_3')
sheet12=book.add_sheet('w2_2_4')
sheet13=book.add_sheet('w2_2_5')
sheet14=book.add_sheet('w2_3_1')
sheet15=book.add_sheet('w2_3_2')
sheet16=book.add_sheet('w2_3_3')
sheet17=book.add_sheet('w2_3_4')
sheet18=book.add_sheet('w2_3_5')
sheet19=book.add_sheet('w3')
sheet20=book.add_sheet('b')
for i in range (5):
    for j in range (5):
        r=i*5
        sheet1.write(r+j,0,w1[i,j,0,0])
        sheet2.write(r+j,0,w1[i,j,0,1])
        sheet3.write(r+j,0,w1[i,j,0,2])
for i in range (5):
    for j in range (5):
        r=i*5
        sheet4.write(r+j,0,w2[i,j,0,0])
        sheet5.write(r+j,0,w2[i,j,0,1])
        sheet6.write(r+j,0,w2[i,j,0,2])
        sheet7.write(r+j,0,w2[i,j,0,3])
        sheet8.write(r+j,0,w2[i,j,0,4])
        sheet9.write(r+j,0,w2[i,j,1,0])
        sheet10.write(r+j,0,w2[i,j,1,1])
        sheet11.write(r+j,0,w2[i,j,1,2])
        sheet12.write(r+j,0,w2[i,j,1,3])
        sheet13.write(r+j,0,w2[i,j,1,4])
        sheet14.write(r+j,0,w2[i,j,2,0])
        sheet15.write(r+j,0,w2[i,j,2,1])
        sheet16.write(r+j,0,w2[i,j,2,2])
        sheet17.write(r+j,0,w2[i,j,2,3])
        sheet18.write(r+j,0,w2[i,j,2,4])
for i in range (125):
    for j in range (10):
        sheet19.write(i,j,w3[i,j])
for i in range (10):
        sheet20.write(i,0,b[i])
book.save('parameters.xls')