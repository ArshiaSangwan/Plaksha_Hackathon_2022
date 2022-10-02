#!/usr/bin/env python
# coding: utf-8

# In[27]:


import tensorflow as tf
import cv2


# In[28]:


import numpy as np # linear algebra
import pandas as pd
from PIL import Image


# In[29]:


# import the libraries #
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Flatten,MaxPooling2D,Dropout
from tensorflow import keras
# import the image data generator 
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[30]:


keras.backend.clear_session()
np.random.seed(42)


# In[31]:


IMAGE_SHAPE = (339,376, 3)


# In[32]:


model = Sequential()

# convolutional and max pool layer #
model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',strides=(1,1),
                activation='relu',input_shape=IMAGE_SHAPE))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',strides=(1,1),
                activation='relu',input_shape=IMAGE_SHAPE))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same',strides=(1,1),
                activation='relu',input_shape=IMAGE_SHAPE))
model.add(MaxPooling2D(pool_size=(2,2)))

# flatten the layer before feeding into the dense layer #
model.add(Flatten())

# dense layer together with dropout to prevent overfitting #
model.add(Dense(units=128,activation='relu',kernel_initializer='he_normal'))
model.add(Dense(units=64,activation='relu',kernel_initializer='he_normal'))
model.add(Dense(units=32,activation='relu',kernel_initializer='he_normal'))
model.add(Dropout(0.5))

# there are 33 classes, hence 33 neurons in the final layer #
model.add(Dense(units=5,activation='softmax'))

# compile the model #
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[33]:


model.summary()


# In[34]:



model = tf.keras.models.load_model("model")


# In[35]:


model.summary()


# In[75]:


image = cv2.imread("chair.jpg", 1)
image = cv2.resize(image, (339, 376))
image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


# In[76]:


from tensorflow.keras.preprocessing.image import img_to_array
test_image_array = img_to_array(image)


# In[77]:


test_image_array.shape


# In[78]:


test_image_array = test_image_array.reshape((-1, 339, 376, 3))


# In[79]:


test_image_array


# In[80]:


pred = model.predict(test_image_array)


# In[81]:


pred


# In[82]:


predicted_class_indices=np.argmax(pred)


# In[83]:


predicted_class_indices


# In[ ]:




