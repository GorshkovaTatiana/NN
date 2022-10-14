#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf



import pathlib

data_dir = r'C:\Users\Татьяна\Desktop\Нейронные сети\flowers'

batch_size = 32
img_height = 150
img_width = 150

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# In[16]:


val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# In[17]:


class_names = train_ds.class_names
print(class_names)


# In[18]:


for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break
      


# In[19]:


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# In[6]:


num_classes = 5
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


num_classes = 5

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(256, activation='relu'),
  layers.Dense(num_classes)
])


# In[22]:


model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])


# In[23]:


model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=10   
)


# In[31]:


from PIL import Image
 
image = Image.open('7.jpg')
image.load()
image.show()


# In[33]:


#конвертация любой картинки в тензор для дальнейшей обработки

image = image.resize((150, 150))
image_tensor = tf.convert_to_tensor(np.array(image), dtype=tf.float32) 
image_tensor = tf.expand_dims(image_tensor, 0)
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
#предсказание 
predictions = probability_model.predict(
        image_tensor, use_multiprocessing=True)
predictions[0]
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(predictions[0])], 100 * np.max(predictions[0]))
)


# In[20]:


#попытка расширить данные и увеличить тем самым точность
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)


# In[21]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),   
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),   
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])


# In[ ]:


test_loss, test_acc = model.evaluate(val_ds, verbose=2)

print('\nTest accuracy:', test_acc)

