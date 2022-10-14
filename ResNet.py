#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastai.vision.all import *
path = r"C:\Users\Татьяна\Desktop\Нейронные сети\flowers"


# In[2]:


files=get_image_files(path)


# In[3]:


files


# In[3]:



flowers=DataBlock(blocks=(ImageBlock,CategoryBlock),
                 get_items=get_image_files,
                 splitter=RandomSplitter(),
                 get_y=parent_label,
                 item_tfms=Resize(224, method=ResizeMethod.Pad))


# In[6]:


loader=flowers.dataloaders(path)
loader.show_batch(max_n=9)


# In[7]:


learn = vision_learner(loader, resnet34, metrics=accuracy)


# In[8]:


learn.fine_tune(10)


# In[9]:


learn.show_results()


# In[21]:


learn.predict(item=r"C:\Users\Татьяна\1.jpg")


# In[21]:


learn.predict(item=r"C:\Users\Татьяна\3.jpg")


# In[22]:


learn.predict(item=r"C:\Users\Татьяна\4.jpg")


# In[23]:


learn.predict(item=r"C:\Users\Татьяна\3.jpg")


# In[24]:


learn.predict(item=r"C:\Users\Татьяна\2.jpg")


# In[10]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# In[26]:


interp.plot_top_losses (6, nrows = 3)


# In[30]:


learn.predict(item=r"C:\Users\Татьяна\6.jpg")


# In[ ]:




