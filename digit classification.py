#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


mnist = tf.keras.datasets.mnist


# In[4]:


(x_train,y_train),(x_test,y_test) = mnist.load_data()


# In[5]:


plt.imshow(x_train[4],cmap=plt.cm.binary)
plt.show()


# In[6]:


x_train[0]


# In[7]:


x_train= tf.keras.utils.normalize(x_train,axis=1)
x_test= tf.keras.utils.normalize(x_test,axis=1)


# In[8]:


x_train[0]


# In[9]:


model=tf.keras.models.Sequential()


# In[10]:


model.add(tf.keras.layers.Flatten())


# In[11]:


model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))


# In[12]:


model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))


# In[13]:


model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))


# In[14]:


model.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy', #how will we calculate the error to minimize the loss
              metrics=['accuracy'])


# In[15]:


model.fit(x_train,y_train,epochs=10)


# In[16]:


val_loss,val_acc=model.evaluate(x_test,y_test)


# In[17]:


val_loss


# In[18]:


val_acc


# In[19]:


model.save(r'C:\Python37\Projects\Digit Recognition\digit_model.model')


# In[20]:


model.save(r'E:\chandu clg\digit_classification')


# In[21]:


new_model=tf.keras.models.load_model(r'C:\Python37\Projects\Digit Recognition\digit_model.model')
predictions=new_model.predict(x_test)


# In[22]:


predictions[0]


# In[23]:


import numpy as np


# In[26]:


plt.imshow(x_test[50],cmap=plt.cm.binary)
plt.show()


# In[29]:


np.argmax(predictions[50])


# In[ ]:




