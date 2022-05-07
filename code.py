%matplotlib inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


!wget --no-check-certificate \
    https://github.com/yenlung/Deep-Learning-Basics/raw/master/images/myna.zip \
    -O /content/myna.zip
    
    
import os
import zipfile

local_zip = '/content/myna.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/content')
zip_ref.close()


from google.colab import drive
drive.mount('/content/drive')


base_dir = '/content/'


myna_folders = ['crested_myna', 'javan_myna', 'common_myna']


thedir = base_dir + myna_folders[0]
os.listdir(thedir)


data = []
target = []

for i in range(3):
    thedir = base_dir + myna_folders[i]
    myna_fnames = os.listdir(thedir)
    for myna in myna_fnames:
        img = load_img(thedir + '/' + myna, target_size = (256,256))
        x = np.array(img)
        data.append(x)
        target.append(i)


data = np.array(data)


x_train = preprocess_input(data)


plt.axis('off')
plt.imshow(x_train[n]);


y_train = to_categorical(target, 3)


y_train[n]


resnet = ResNet50V2(include_top=False, pooling="avg")


resnet.trainable = False


model = Sequential()


model.add(resnet)


model.add(Dense(3, activation='softmax'))


model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
              
              
model.summary()             


model.fit(x_train, y_train, batch_size=23, epochs=10)


y_predict = np.argmax(model.predict(x_train), -1)


labels = ["土八哥", "白尾八哥", "家八哥"]


!pip install gradio


import gradio as gr


def classify_image(inp):
  inp = inp.reshape((-1, 256, 256, 3))
  inp = preprocess_input(inp)
  prediction = model.predict(inp).flatten()
  return {labels[i]: float(prediction[i]) for i in range(3)}

image = gr.inputs.Image(shape=(256, 256), label="八哥照片")
label = gr.outputs.Label(num_top_classes=3, label="所以八哥是")


gr.Interface(fn=classify_image, inputs=image, outputs=label,
             title="AI 八哥辨識機",
             description="我能辨識台灣常見的三種八哥: (土)八哥、家八哥、白尾八哥。快找張八哥照片來考我吧!",
             capture_session=True).launch()
