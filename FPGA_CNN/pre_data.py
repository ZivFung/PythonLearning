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
XTrain = np.zeros((SampleNum*10,32,32))
YTrain = np.zeros((SampleNum*10,10))

for i in range(StartNum,EndNum):
    XTrain[(i-StartNum),:,:] = np.array(Image.open('F:/Study/Cnn/erzhi/Figure32/zero' + str(i) + '.jpg'))
    XTrain[XTrain > threhold ] = 1
    YTrain[(i-StartNum),:] = [1,0,0,0,0,0,0,0,0,0]
for i in range(StartNum,EndNum):
    XTrain[(i+SampleNum-StartNum),:,:] = np.array(Image.open('F:/Study/Cnn/erzhi/Figure32/one' + str(i) + '.jpg'))
    XTrain[XTrain > threhold ] = 1
    YTrain[(i+SampleNum-StartNum),:] =  [0,1,0,0,0,0,0,0,0,0]
for i in range(StartNum,EndNum):
    XTrain[(i+SampleNum*2-StartNum),:,:] = np.array(Image.open('F:/Study/Cnn/erzhi/Figure32/two' + str(i) + '.jpg'))
    XTrain[XTrain > threhold ] = 1
    YTrain[(i+SampleNum*2-StartNum),:] =  [0,0,1,0,0,0,0,0,0,0]
for i in range(StartNum,EndNum):
    XTrain[(i+SampleNum*3-StartNum),:,:] = np.array(Image.open('F:/Study/Cnn/erzhi/Figure32/three' + str(i) + '.jpg'))
    XTrain[XTrain > threhold ] = 1
    YTrain[(i+SampleNum*3-StartNum),:] = [0,0,0,1,0,0,0,0,0,0]
for i in range(StartNum,EndNum):
    XTrain[(i+SampleNum*4-StartNum),:,:] = np.array(Image.open('F:/Study/Cnn/erzhi/Figure32/four' + str(i) + '.jpg'))
    XTrain[XTrain > threhold ] = 1
    YTrain[(i+SampleNum*4-StartNum),:] = [0,0,0,0,1,0,0,0,0,0]
for i in range(StartNum,EndNum):
    XTrain[(i+SampleNum*5-StartNum),:,:] = np.array(Image.open('F:/Study/Cnn/erzhi/Figure32/five' + str(i) + '.jpg'))
    XTrain[XTrain > threhold ] = 1
    YTrain[(i+SampleNum*5-StartNum),:] = [0,0,0,0,0,1,0,0,0,0]
for i in range(StartNum,EndNum):
    XTrain[(i+SampleNum*6-StartNum),:,:] = np.array(Image.open('F:/Study/Cnn/erzhi/Figure32/six' + str(i) + '.jpg'))
    XTrain[XTrain > threhold ] = 1
    YTrain[(i+SampleNum*6-StartNum),:] = [0,0,0,0,0,0,1,0,0,0]
for i in range(StartNum,EndNum):
    XTrain[(i+SampleNum*7-StartNum),:,:] = np.array(Image.open('F:/Study/Cnn/erzhi/Figure32/seven' + str(i) + '.jpg'))
    XTrain[XTrain > threhold ] = 1
    YTrain[(i+SampleNum*7-StartNum),:] = [0,0,0,0,0,0,0,1,0,0]
for i in range(StartNum,EndNum):
    XTrain[(i+SampleNum*8-StartNum),:,:] = np.array(Image.open('F:/Study/Cnn/erzhi/Figure32/eight' + str(i) + '.jpg'))
    XTrain[XTrain > threhold ] = 1
    YTrain[(i+SampleNum*8-StartNum),:] = [0,0,0,0,0,0,0,0,1,0]
for i in range(StartNum,EndNum):
    XTrain[(i+SampleNum*9-StartNum),:,:] = np.array(Image.open('F:/Study/Cnn/erzhi/Figure32/nine' + str(i) + '.jpg'))
    XTrain[XTrain > threhold ] = 1
    YTrain[(i+SampleNum*9-StartNum),:] = [0,0,0,0,0,0,0,0,0,1]
XTrain = XTrain[:,:,:,np.newaxis]
print(XTrain.shape,YTrain.shape)
# (15,32,32,1) (15,4)

XTest = np.zeros((TrainNum*10,32,32))
YTest = np.zeros((TrainNum*10,10))
for i in range(EndNum+1,EndNum+TrainNum):
    XTest[(i-(EndNum+1)),:,:] = np.array(Image.open('F:/Study/Cnn/erzhi/Figure32/zero' + str(i) + '.jpg'))
    XTest[XTest > threhold] = 1
    YTest[(i-(EndNum+1)),:] = [1,0,0,0,0,0,0,0,0,0]
for i in range(EndNum+1,EndNum+TrainNum):
    XTest[(i-(EndNum+1-TrainNum)),:,:] = np.array(Image.open('F:/Study/Cnn/erzhi/Figure32/one' + str(i) + '.jpg'))
    XTest[XTest > threhold] = 1
    YTest[(i-(EndNum+1-TrainNum)),:] = [0,1,0,0,0,0,0,0,0,0]
for i in range(EndNum+1,EndNum+TrainNum):
    XTest[(i-(EndNum+1-TrainNum*2)),:,:] = np.array(Image.open('F:/Study/Cnn/erzhi/Figure32/two' + str(i) + '.jpg'))
    XTest[XTest > threhold] = 1
    YTest[(i-(EndNum+1-TrainNum*2)),:] = [0,0,1,0,0,0,0,0,0,0]
for i in range(EndNum+1,EndNum+TrainNum):
    XTest[(i-(EndNum+1-TrainNum*3)),:,:] = np.array(Image.open('F:/Study/Cnn/erzhi/Figure32/three' + str(i) + '.jpg'))
    XTest[XTest > threhold] = 1
    YTest[(i-(EndNum+1-TrainNum*3)),:] = [0,0,0,1,0,0,0,0,0,0]
for i in range(EndNum+1,EndNum+TrainNum):
    XTest[(i-(EndNum+1-TrainNum*4)),:,:] = np.array(Image.open('F:/Study/Cnn/erzhi/Figure32/four' + str(i) + '.jpg'))
    XTest[XTest > threhold] = 1
    YTest[(i-(EndNum+1-TrainNum*4)),:] = [0,0,0,0,1,0,0,0,0,0]
for i in range(EndNum+1,EndNum+TrainNum):
    XTest[(i-(EndNum+1-TrainNum*5)),:,:] = np.array(Image.open('F:/Study/Cnn/erzhi/Figure32/five' + str(i) + '.jpg'))
    XTest[XTest > threhold] = 1
    YTest[(i-(EndNum+1-TrainNum*5)),:] = [0,0,0,0,0,1,0,0,0,0]
for i in range(EndNum+1,EndNum+TrainNum):
    XTest[(i-(EndNum+1-TrainNum*6)),:,:] = np.array(Image.open('F:/Study/Cnn/erzhi/Figure32/six' + str(i) + '.jpg'))
    XTest[XTest > threhold] = 1
    YTest[(i-(EndNum+1-TrainNum*6)),:] = [0,0,0,0,0,0,1,0,0,0]
for i in range(EndNum+1,EndNum+TrainNum):
    XTest[(i-(EndNum+1-TrainNum*7)),:,:] = np.array(Image.open('F:/Study/Cnn/erzhi/Figure32/seven' + str(i) + '.jpg'))
    XTest[XTest > threhold] = 1
    YTest[(i-(EndNum+1-TrainNum*7)),:] = [0,0,0,0,0,0,0,1,0,0]
for i in range(EndNum+1,EndNum+TrainNum):
    XTest[(i-(EndNum+1-TrainNum*8)),:,:] = np.array(Image.open('F:/Study/Cnn/erzhi/Figure32/eight' + str(i) + '.jpg'))
    XTest[XTest > threhold] = 1
    YTest[(i-(EndNum+1-TrainNum*8)),:] = [0,0,0,0,0,0,0,0,1,0]
for i in range(EndNum+1,EndNum+TrainNum):
    XTest[(i-(EndNum+1-TrainNum*9)),:,:] = np.array(Image.open('F:/Study/Cnn/erzhi/Figure32/nine' + str(i) + '.jpg'))
    XTest[XTest > threhold] = 1
    YTest[(i-(EndNum+1-TrainNum*9)),:] = [0,0,0,0,0,0,0,0,0,1]
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

