# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 16:17:53 2019

@author: Yen_Wei
"""


from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Flatten,MaxPooling2D,Dropout,Conv2D,Dense
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(10)

def show_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2,4)
    plt.imshow(image,cmap='binary')
    plt.show()

def show_images_label_predictions(images,labels,predictions,start_id,num=10):
    plt.gcf().set_size_inches(12,14)
    if num>25:
        num=25
    for i in range(0,num):
        ax = plt.subplot(5, 5,1+i)
        ax.imshow(images[start_id],cmap='binary')  #顯示黑白
        
        #有預測結果才在標題顯現
        if(len(predictions) > 0):
            title = 'ai=' + str(predictions[i])
            title += ('(o)' if predictions[i]==labels[i] else '(x)')
            title += '\nlabel=' + str(labels[i])
        else:
            title = 'label=' + str(labels[i])
        
        ax.set_title(title,fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        start_id+=1
        
    plt.show()

(train_feature,train_label),(test_feature,test_label) = mnist.load_data()

train_feature_vector = train_feature.reshape(train_feature.shape[0],28,28,1).astype('float32')
test_feature_vector = test_feature.reshape(test_feature.shape[0],28,28,1).astype('float32')

train_feature_normalize = train_feature_vector/255
test_feature_normalize = test_feature_vector/255

train_label_onehot = np_utils.to_categorical(train_label)
test_label_onehot = np_utils.to_categorical(test_label)

model = Sequential()

model.add(Conv2D(input_shape=(28,28,1),filters=16,kernel_size=(5,5),activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(filters=36,kernel_size=(5,5),activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(256,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

train_history = model.fit(x=train_feature_normalize,y=train_label_onehot,validation_split=0.2,epochs=10,batch_size=300)

scores = model.evaluate(x=train_feature_normalize,y=train_label_onehot)
print("Training Acc:",scores[1])

model.save('Minst_CNN_model.h5')
print("模型儲存完畢")
#將參數儲存不包含模型
model.save_weights("Minst_CNN_model2.weight")
print("參數儲存完畢")

prediction = model.predict_classes(test_feature_normalize)

show_images_label_predictions(test_feature, test_label, prediction, 0)