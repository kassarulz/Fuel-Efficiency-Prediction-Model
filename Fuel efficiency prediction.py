#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from keras import regularizers

print(tf.__version__)


# In[2]:


dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
dataset_path


# In[3]:


column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.head(20)


# In[4]:


dataset.isna().sum()


# In[5]:


dataset.columns


# In[6]:


ptable = pd.pivot_table(dataset, index=["Cylinders", "Displacement"], values=["MPG"], aggfunc='count')
ptable.plot()


# In[7]:


ptable = pd.pivot_table(dataset, index=["Cylinders", "Displacement"], columns = ["Horsepower"], values=["MPG"], dropna=True, fill_value=0, aggfunc=np.sum)
ptable.plot()


# In[8]:


ptable = pd.pivot_table(dataset, index=["Cylinders", "Displacement"], values=["MPG"])
ptable


# In[9]:


dataset = dataset.dropna()


# In[10]:


dataset['Origin'] = dataset['Origin'].map(lambda x: {1: 'USA', 2: 'Europe', 3: 'Japan'}.get(x))


# In[11]:


dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
dataset.head(20)


# In[12]:


train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)


# In[13]:


sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")


# In[14]:


train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
train_stats


# In[15]:


train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')


# In[16]:


def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


# In[17]:


train_dataset


# # Ridge Rigression

# In[18]:


y = train_labels


# In[19]:


X = normed_train_data.copy()


# In[20]:


X['bias'] = 1


# In[21]:


X.head()


# In[22]:


delta = 1


# In[23]:


I = np.identity(X.shape[1], dtype = float) 


# In[24]:


X.transpose().dot(X).shape


# In[25]:


reg_estimated_param = np.linalg.pinv(X.transpose().dot(X) + pow(delta,2)*I ).dot(X.transpose().dot(y))


# In[26]:


reg_estimated_param


# In[27]:


def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model


# In[28]:


def build_model_regularized():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())], kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model


# In[29]:


model = build_model()


# In[30]:


model.summary()


# In[31]:


example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
example_result


# In[32]:


# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])


# In[33]:


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


# In[34]:


def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'val_loss')
  plt.ylim([0,5])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'val_loss')
  plt.ylim([0,20])
  plt.legend()
  plt.show()


plot_history(history)


# In[35]:


model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)


# In[36]:


loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))


# # Training after regularization

# In[37]:


reg_model = build_model_regularized()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = reg_model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])


# In[38]:


plot_history(history)


# In[39]:


reg_loss, reg_mae, reg_mse = reg_model.evaluate(normed_test_data, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))


# # Predictions

# In[40]:


test_predictions = model.predict(normed_test_data).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# In[41]:


linear_error = test_predictions - test_labels
plt.hist(linear_error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")


# In[42]:


linear_error_sum = abs(linear_error).sum()


# ### Predictions with Regularization

# In[43]:


reg_linear_test_predictions = reg_model.predict(normed_test_data).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, reg_linear_test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# In[44]:


reg_linear_error = reg_linear_test_predictions - test_labels
plt.hist(linear_error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")


# In[45]:


reg_linear_error_sum = abs(reg_linear_error).sum()


# ### Make predictions Ridge Regression

# In[46]:


normed_test_data['bias'] = 1.0


# In[47]:


ridge_test_predictions = normed_test_data.dot(reg_estimated_param.transpose())

a = plt.axes(aspect='equal')
plt.scatter(test_labels, ridge_test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# In[48]:


ridge_error = ridge_test_predictions - test_labels
plt.hist(ridge_error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")


# In[49]:


ridge_error_sum = abs(ridge_error).sum()


# In[50]:


# Initilize the list
data = [['Linear', linear_error_sum], ['Regularized', reg_linear_error_sum], ['Ridge', ridge_error_sum]] 
   
df = pd.DataFrame(data, columns = ['Name', 'Error']) 
  
df 


# In[ ]:




