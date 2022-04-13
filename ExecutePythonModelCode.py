
# coding: utf-8

# In[ ]:


import numpy as np
import keras
from keras import backend as K
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import keras
import time
import os.path

import matplotlib.pyplot as plt


# In[ ]:


def createInputFile(X):

    fInputImage = open("/media/nikhil/New Volume/Works/Cuda Work/MobileNets/MobileNets/MobileNets/data/FirstLayer/InputFiles/inputsNorm.txt", "w")
    for i in range(len(X[0][0])):
        for j in range(len(X)):
            for k in range(len(X[0])):
                fInputImage.write(str(X[j][k][i]) + "\n")

    fInputImage.close()
    print("Input File writing complete!!!")


# In[ ]:


def prepare_image(file):
    img = image.load_img(file, target_size=(224,224))
    img_array = image.img_to_array(img)
    W = np.array(img_array)
    print(W.shape)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


# In[10]:


fileName = "Dog.jpg"
preprocessed_image = prepare_image(fileName)
createInputFile(preprocessed_image[0])

mobile = keras.applications.mobilenet.MobileNet()
print(preprocessed_image.shape)
predictions = mobile.predict(preprocessed_image)
results = imagenet_utils.decode_predictions(predictions)
print(results)


# In[11]:


print(predictions.shape)
print(len(predictions[0]))
fOut = open("PyOutput.txt", "w")
for i in range(len(predictions[0])):
    fOut.write(str(predictions[0][i]) + "\n")
fOut.close()


# In[12]:


print("Python code prediction --> \n\n")
for i in range(len(results[0])):
    print(results[0][i][1])


# In[13]:


print(type(predictions))
print(predictions.shape)


# In[14]:

"""
import subprocess
subprocess.run(["nvcc", "MobileNets_host.cu","-o","MN"])
subprocess.run(["./MN"])


# In[15]:


fCudaOutput = open("/media/nikhil/New Volume/Works/Cuda Work/MobileNets/MobileNets/data/TwentyNineLayer/output.txt", "r")
lines = fCudaOutput.readlines()
nCudaOutput = np.array(lines)
print(nCudaOutput.shape)
nCudaOutput = nCudaOutput.reshape([1,1000])
print(nCudaOutput.shape)
results = imagenet_utils.decode_predictions(nCudaOutput)
print("Cuda code prediction --> \n\n")
for i in range(len(results[0])):
    print(results[0][i][1])
    
fCudaOutput.close()

"""