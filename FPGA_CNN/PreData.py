import numpy as np
# import cv2 
import math
import h5py
from PIL import Image

StartNum = 1
EndNum = 190
SampleNum = EndNum - StartNum + 1

TrainNum = 10
threhold = 100
XTrain = np.zeros((1900,32,32))
YTrain = np.zeros((1900,10))

for i in range(1,190):
    XTrain[i-1,:,:] = np.array(Image.open('F:/Study/Cnn/erzhi/Bool32/zero' + str(i) + '.bmp'))
    XTrain[XTrain > threhold ] = 1
    YTrain[i-1,:] = [1,0,0,0,0,0,0,0,0,0]
for i in range(1,190):
    XTrain[i+190-1,:,:] = np.array(Image.open('F:/Study/Cnn/erzhi/Bool32/one' + str(i) + '.bmp'))
    XTrain[XTrain > threhold ] = 1
    YTrain[i+190-1,:] =  [0,1,0,0,0,0,0,0,0,0]
for i in range(1,190):
    XTrain[i+380-1,:,:] = np.array(Image.open('F:/Study/Cnn/erzhi/Bool32/two' + str(i) + '.bmp'))
    XTrain[XTrain > threhold ] = 1
    YTrain[i+380-1,:] =  [0,0,1,0,0,0,0,0,0,0]
for i in range(1,190):
    XTrain[i+570-1,:,:] = np.array(Image.open('F:/Study/Cnn/erzhi/Bool32/three' + str(i) + '.bmp'))
    XTrain[XTrain > threhold ] = 1
    YTrain[i+570-1,:] = [0,0,0,1,0,0,0,0,0,0]
for i in range(1,190):
    XTrain[i+760-1,:,:] = np.array(Image.open('F:/Study/Cnn/erzhi/Bool32/four' + str(i) + '.bmp'))
    XTrain[XTrain > threhold ] = 1
    YTrain[i+760-1,:] = [0,0,0,0,1,0,0,0,0,0]
for i in range(1,190):
    XTrain[i+950-1,:,:] = np.array(Image.open('F:/Study/Cnn/erzhi/Bool32/five' + str(i) + '.bmp'))
    XTrain[XTrain > threhold ] = 1
    YTrain[i+950-1,:] = [0,0,0,0,0,1,0,0,0,0]
for i in range(1,190):
    XTrain[i+1140-1,:,:] = np.array(Image.open('F:/Study/Cnn/erzhi/Bool32/six' + str(i) + '.bmp'))
    XTrain[XTrain > threhold ] = 1
    YTrain[i+1140-1,:] = [0,0,0,0,0,0,1,0,0,0]
for i in range(1,190):
    XTrain[i+1330-1,:,:] = np.array(Image.open('F:/Study/Cnn/erzhi/Bool32/seven' + str(i) + '.bmp'))
    XTrain[XTrain > threhold ] = 1
    YTrain[i+1330-1,:] = [0,0,0,0,0,0,0,1,0,0]
for i in range(1,190):
    XTrain[i+1520-1,:,:] = np.array(Image.open('F:/Study/Cnn/erzhi/Bool32/eight' + str(i) + '.bmp'))
    XTrain[XTrain > threhold ] = 1
    YTrain[i+1520-1,:] = [0,0,0,0,0,0,0,0,1,0]
for i in range(1,190):
    XTrain[i+1710-1,:,:] = np.array(Image.open('F:/Study/Cnn/erzhi/Bool32/nine' + str(i) + '.bmp'))
    XTrain[XTrain > threhold ] = 1
    YTrain[i+1710-1,:] = [0,0,0,0,0,0,0,0,0,1]
XTrain = XTrain[:,:,:,np.newaxis]
print(XTrain.shape,YTrain.shape)
# (15,32,32,1) (15,4)

XTest = np.zeros((100,32,32))
YTest = np.zeros((100,10))
for i in range(191,200):
    XTest[i-191,:,:] = np.array(Image.open('F:/Study/Cnn/erzhi/Bool32/zero' + str(i) + '.bmp'))
    XTest[XTest > threhold] = 1
    YTest[i-191,:] = [1,0,0,0,0,0,0,0,0,0]
for i in range(191,200):
    XTest[i-191+10,:,:] = np.array(Image.open('F:/Study/Cnn/erzhi/Bool32/one' + str(i) + '.bmp'))
    XTest[XTest > threhold] = 1
    YTest[i-191+10,:] = [0,1,0,0,0,0,0,0,0,0]
for i in range(191,200):
    XTest[i-191+20,:,:] = np.array(Image.open('F:/Study/Cnn/erzhi/Bool32/two' + str(i) + '.bmp'))
    XTest[XTest > threhold] = 1
    YTest[i-191+20,:] = [0,0,1,0,0,0,0,0,0,0]
for i in range(191,200):
    XTest[i-191+30,:,:] = np.array(Image.open('F:/Study/Cnn/erzhi/Bool32/three' + str(i) + '.bmp'))
    XTest[XTest > threhold] = 1
    YTest[i-191+30,:] = [0,0,0,1,0,0,0,0,0,0]
for i in range(191,200):
    XTest[i-191+40,:,:] = np.array(Image.open('F:/Study/Cnn/erzhi/Bool32/four' + str(i) + '.bmp'))
    XTest[XTest > threhold] = 1
    YTest[i-191+40,:] = [0,0,0,0,1,0,0,0,0,0]
for i in range(191,200):
    XTest[i-191+50,:,:] = np.array(Image.open('F:/Study/Cnn/erzhi/Bool32/five' + str(i) + '.bmp'))
    XTest[XTest > threhold] = 1
    YTest[i-191+50,:] = [0,0,0,0,0,1,0,0,0,0]
for i in range(191,200):
    XTest[i-191+60,:,:] = np.array(Image.open('F:/Study/Cnn/erzhi/Bool32/six' + str(i) + '.bmp'))
    XTest[XTest > threhold] = 1
    YTest[i-191+60,:] = [0,0,0,0,0,0,1,0,0,0]
for i in range(91,100):
    XTest[i-191+70,:,:] = np.array(Image.open('F:/Study/Cnn/erzhi/Bool32/seven' + str(i) + '.bmp'))
    XTest[XTest > threhold] = 1
    YTest[i-191+70,:] = [0,0,0,0,0,0,0,1,0,0]
for i in range(191,200):
    XTest[i-191+80,:,:] = np.array(Image.open('F:/Study/Cnn/erzhi/Bool32/eight' + str(i) + '.bmp'))
    XTest[XTest > threhold] = 1
    YTest[i-191+80,:] = [0,0,0,0,0,0,0,0,1,0]
for i in range(191,200):
    XTest[i-191+90,:,:] = np.array(Image.open('F:/Study/Cnn/erzhi/Bool32/nine' + str(i) + '.bmp'))
    XTest[XTest > threhold] = 1
    YTest[i-191+90,:] = [0,0,0,0,0,0,0,0,0,1]
# for i in range(12,16):
#     XTest[(i-8),:,:] = np.array(Image.open('' + str(i) + '.jpg'))
#     XTest[XTest > 0] = 1
#     YTest[(i-8),:] = [0,0,1,0]
XTest = XTest[:,:,:,np.newaxis]
print(XTest.shape,YTest.shape)
# (8,32,32,1) (4,4)


File = h5py.File('./Dataset3_number.h5','w')
File.create_dataset('train_set_x',data = XTrain)
File.create_dataset('train_set_y',data = YTrain)
File.create_dataset('test_set_x',data = XTest)
File.create_dataset('test_set_y',data = YTest)
File.close()

