#!/usr/bin/env python
# coding: utf-8

# In[1]:


import statistics as st
import math
import pprint as pr
import pandas as pd
import numpy as np
import datetime
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

meteo_egnatia=pd.read_excel('METEOROLOGICAL_EGNATIA_2013-2017.xlsx',sheet_name='DATA')
meteo_eptapyrgio=pd.read_excel('METEOROLOGICAL_EPTAPYRGIO_2013-2017.xlsx',sheet_name='DATA')
poll_egnatia1=pd.read_excel('Pollution_2010_2013_version_1.xlsx',sheet_name='Στ. ΕΓΝΑΤΙΑΣ')
poll_egnatia2=pd.read_excel('Pollution_2014_2016_version_1.xlsx',sheet_name='Στ. ΕΓΝΑΤΙΑΣ')


# # Δημιουργία αρχείου μετεωρολογικών χαρακτηριστικών και αρχείου PM10
# Μετατροπή σε αριθμητικά δεδομένα και τοποθέτηση NaN όπου βρίσκει κενό (errors='coerse')

# In[2]:


meteo_data=[]
meteo_data=pd.DataFrame(meteo_data)
meteo_data['Date']=meteo_egnatia.loc[7:,'Station:']
meteo_data['WS-V']=meteo_eptapyrgio.loc[7:,'Unnamed: 2']
meteo_data['WD-V']=meteo_eptapyrgio.loc[7:,'Unnamed: 3']
meteo_data['Tout']=meteo_egnatia.loc[7:,'Unnamed: 4']
meteo_data['RH']=meteo_egnatia.loc[7:,'Unnamed: 5']
meteo_data['Time']=meteo_eptapyrgio.loc[7:,'EPTAPYRGIO']
meteo_data=meteo_data.reset_index(drop=True)
meteo_data['WS-V']=pd.to_numeric(meteo_data['WS-V'],errors='coerce')
meteo_data['WD-V']=pd.to_numeric(meteo_data['WD-V'],errors='coerce')
meteo_data['Tout']=pd.to_numeric(meteo_data['Tout'],errors='coerce')
meteo_data['RH']=pd.to_numeric(meteo_data['RH'],errors='coerce')
meteo_data['Date']=pd.to_datetime(meteo_data['Date'])


# In[3]:


poll_data=[]
poll_data=pd.DataFrame(poll_data)
poll_data['Date']=poll_egnatia1.loc[:,'Ημερο -\nμηνία'].append(poll_egnatia2.loc[:,'Ημερο -\nμηνία'],
                                                               ignore_index=True)
poll_data['PM10']=poll_egnatia1.loc[:,'PM10\nμg/m3'].append(poll_egnatia2.loc[:,'PM10\nμg/m3'],ignore_index=True)
poll_data['PM10']=pd.to_numeric(poll_data['PM10'],errors='coerce')


# # Μετατροπή διεύθυνσης ανέμου σε ημιτονοειδή μορφή

# In[4]:


meteo_data['WD-V']=np.sin(meteo_data['WD-V']*np.pi/180)    #Convert to radians


# In[5]:


meteo_data


# # Διαχωρισμός προηγούμενου αρχείου σε περιόδους '13-'15 και '16
# Ο διαχωρισμός έγινε με βάση την ημερομηνία και τα δύο αρχεία συνενώθηκαν για τη δημιουργία του training_set και του test_set.

# In[6]:


meteo_data1315=[]
meteo_data1315=pd.DataFrame(meteo_data1315)
meteo1315_temp1=[]
meteo1315_temp1=pd.DataFrame(meteo1315_temp1)
meteo1315_temp2=[]
meteo1315_temp2=pd.DataFrame(meteo1315_temp2)
meteo1315=[]
meteo1315=pd.DataFrame(meteo1315)

start_date_meteo13=list(meteo_data[meteo_data['Date']=='2013/1/1'].index.values)
end_date_meteo15=list(meteo_data[meteo_data['Date']=='2015/12/31'].index.values)

meteo_data1315=meteo_data.iloc[start_date_meteo13[0]:end_date_meteo15[-1]+1]

time_wind1315=list(meteo_data1315[meteo_data1315['Time']=='14:00'].index.values)
meteo1315_temp1[['WS-V','WD-V']]=meteo_data1315[['WS-V','WD-V']].loc[time_wind1315]
meteo1315_temp1.index=np.arange(len(meteo1315_temp1))

meteo1315_temp2=meteo_data1315[['Date','Tout','RH']]
meteo1315_temp2=meteo1315_temp2.groupby('Date')[['Tout','RH']].mean()
meteo1315_temp2=meteo1315_temp2.reset_index(inplace=False)

meteo1315['Date']=meteo1315_temp2['Date']
meteo1315[['WS-V','WD-V']]=meteo1315_temp1[['WS-V','WD-V']]
meteo1315['Tout']=meteo1315_temp2['Tout']
meteo1315['RH']=meteo1315_temp2['RH']
meteo1315


# In[7]:


poll1315=[]
poll1315=pd.DataFrame(poll1315)

start_date_poll13=int(poll_data[poll_data['Date']=='2013/1/1'].index.values)
end_date_poll15=int(poll_data[poll_data['Date']=='2015/12/31'].index.values)

poll1315=poll_data.iloc[start_date_poll13:end_date_poll15+1]
poll1315=poll1315.reset_index(drop=True)


# Για τα δύο σετ κάναμε κλώνο τη στήλη του PM10 και τη μετακινήσαμε μία θέση κάτω, ώστε να έχουμε στην ίδια ημερομηνία και το PM10 της προηγούμενης μέρας για να μπορούμε να ελέγξουμε αν υπάρχουν NaN. Στη συνέχεια πετάξαμε έξω όσες σειρές είχαν έλλειψη έστω και σε μία στήλη.

# In[8]:


training_set=[]
training_set=pd.DataFrame(training_set)
training_set[['Date','WS-V','WD-V','Tout','RH']]=meteo1315[['Date','WS-V','WD-V','Tout','RH']]
training_set['PM10-PD']=poll1315['PM10']
training_set['PM10']=poll1315['PM10']
training_set['PM10-PD']=training_set['PM10-PD'].shift()
training_set=training_set.dropna()
training_set=training_set.reset_index(drop=True)
training_set


# In[9]:


meteo_data16=[]
meteo_data16=pd.DataFrame(meteo_data16)
meteo16_temp1=[]
meteo16_temp1=pd.DataFrame(meteo16_temp1)
meteo16_temp2=[]
meteo16_temp2=pd.DataFrame(meteo16_temp2)
meteo16=[]
meteo16=pd.DataFrame(meteo16)

start_date_meteo16=list(meteo_data[meteo_data['Date']=='2016/1/1'].index.values)
end_date_meteo16=list(meteo_data[meteo_data['Date']=='2016/12/31'].index.values)

meteo_data16=meteo_data.iloc[start_date_meteo16[0]:end_date_meteo16[-1]+1]

time_wind16=list(meteo_data16[meteo_data16['Time']=='14:00'].index.values)
meteo16_temp1[['WS-V','WD-V']]=meteo_data16[['WS-V','WD-V']].loc[time_wind16]
meteo16_temp1.index=np.arange(len(meteo16_temp1))

meteo16_temp2=meteo_data16[['Date','Tout','RH']]
meteo16_temp2=meteo16_temp2.groupby('Date')[['Tout','RH']].mean()
meteo16_temp2=meteo16_temp2.reset_index(inplace=False)

meteo16['Date']=meteo16_temp2['Date']
meteo16[['WS-V','WD-V']]=meteo16_temp1[['WS-V','WD-V']]
meteo16['Tout']=meteo16_temp2['Tout']
meteo16['RH']=meteo16_temp2['RH']


# In[10]:


poll16=[]
poll16=pd.DataFrame(poll16)

start_date_poll16=int(poll_data[poll_data['Date']=='2016/1/1'].index.values)
end_date_poll16=int(poll_data[poll_data['Date']=='2016/12/31'].index.values)

poll16=poll_data.iloc[start_date_poll16:end_date_poll16+1]
poll16=poll16.reset_index(drop=True)


# In[11]:


test_set=[]
test_set=pd.DataFrame(test_set)
test_set[['Date','WS-V','WD-V','Tout','RH']]=meteo16[['Date','WS-V','WD-V','Tout','RH']]
test_set['PM10-PD']=poll16['PM10']
test_set['PM10']=poll16['PM10']
test_set['PM10-PD']=test_set['PM10-PD'].shift()
test_set=test_set.dropna()
test_set=test_set.reset_index(drop=True)
test_set


# In[12]:


total_data=[]
total_data=pd.DataFrame(total_data)
total_data[['Date','WS-V','WD-V','Tout','RH','PM10']]=training_set[['Date','WS-V','WD-V','Tout','RH','PM10']].append(test_set[['Date','WS-V','WD-V','Tout','RH','PM10']],ignore_index=True)


titles = ['Wind Speed','Wind Direction','Temperature','Relative Humidity','Pollution']

colors = ["blue","orange","green","red","olive"]

feature_keys = ['WS-V','WD-V','Tout','RH','PM10']
date_time_key = 'Date'

def show_raw_visualization(data):
    time_data = data[date_time_key]
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(10,15), dpi=80, facecolor="w", edgecolor="k")
    for i in range(len(feature_keys)):
        key = feature_keys[i]
        c = colors[i % (len(colors))]
        t_data = data[key]
        t_data.index = time_data
        t_data.head()
        ax = t_data.plot(
            ax=axes[i],
            color=c,
            title="{} - {}".format(titles[i], key),
            rot=25,
        )
        ax.legend([titles[i]])
    plt.tight_layout()


show_raw_visualization(total_data)


# In[13]:


def show_heatmap(data,title):
    plt.matshow(data.corr())
    plt.xticks(np.arange(data.shape[1]), data.columns, fontsize=14, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(np.arange(data.shape[1]), data.columns, fontsize=14)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title(title, fontsize=14)
    plt.show()
    
show_heatmap(training_set.iloc[:,[1,2,3,4,6]],"Feature Correlation Heatmap\nfor Training Set")
show_heatmap(test_set.iloc[:,[1,2,3,4,6]],"Feature Correlation Heatmap\nfor Test Set")
show_heatmap(pd.concat((training_set.iloc[:,[1,2,3,4,6]],                        test_set.iloc[:,[1,2,3,4,6]])),"Feature Correlation Heatmap")


# # Boxplot

# In[14]:


import seaborn as sns
boxplot_data=[]
boxplot_data=pd.DataFrame(boxplot_data)
boxplot_data[['WS-V','WD-V','Tout','RH','PM10']]=training_set[['WS-V','WD-V','Tout','RH','PM10']]                            .append(test_set[['WS-V','WD-V','Tout','RH','PM10']],ignore_index=True)
boxplot_keys=boxplot_data.keys()
boxplot_data=boxplot_data.melt(var_name='Column', value_name='Normalized')
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalized', data=boxplot_data)
_=ax.set_xticklabels(boxplot_keys, rotation=90)


# # Κανονικοποίηση

# In[15]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

training_set=training_set[['WS-V','WD-V','Tout','RH','PM10-PD','PM10']]
test_set=test_set[['WS-V','WD-V','Tout','RH','PM10-PD','PM10']]

#Normalize
training_set_scaled=((training_set-training_set.mean())/training_set.std()).values
test_set_scaled=((test_set-test_set.mean())/test_set.std()).values

x_train=training_set_scaled[:,0:5].reshape(training_set_scaled[:,0:5]                                           .shape[0],1,training_set_scaled[:,0:5].shape[1])
y_train=training_set_scaled[:,-1].reshape(training_set_scaled[:,-1].shape[0],1)
x_test=test_set_scaled[:,0:5].reshape(test_set_scaled[:,0:5].shape[0],1,test_set_scaled[:,0:5].shape[1])
y_test=test_set_scaled[:,-1].reshape(test_set_scaled[:,-1].shape[0],1)


# # Boxplot to normalized data

# In[16]:


import seaborn as sns

boxplot_Norm_data=np.concatenate((training_set_scaled,test_set_scaled))
boxplot_Norm_data=pd.DataFrame(boxplot_Norm_data,columns=['WS-V','WD-V','Tout','RH','PM10-PD','PM10'])
boxplot_Norm_keys=boxplot_Norm_data.keys()
boxplot_Norm_data=boxplot_Norm_data.melt(var_name='Column', value_name='Normalized')
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalized', data=boxplot_Norm_data)
_=ax.set_xticklabels(boxplot_Norm_keys, rotation=90)


# # Artificial Neural Network

# In[17]:


batch_size=1049
epochs=1000
input_shape=(x_train.shape[1],x_train.shape[2])

def rmse(y_true, y_test):
    from keras import backend
    return backend.sqrt(backend.mean(backend.square(y_test - y_true), axis=-1))
    
def r_square(y_true, y_test):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_test))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


model=Sequential()
model.add(LSTM(100, activation='linear', input_shape=input_shape,return_sequences=True))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_absolute_error',metrics=['mean_squared_error',rmse,r_square]) 
model.summary()


# In[18]:


from keras.callbacks import EarlyStopping

earlystopping=EarlyStopping(monitor='mean_squared_error', patience=40, verbose=1, mode='auto')

model_history=model.fit(
    x_train,
    y_train,
    epochs=epochs,
    validation_data=(x_test,y_test),
    batch_size=batch_size,
    callbacks=[earlystopping]
)


# In[19]:


def visualize_loss(model_history, title):
    loss = model_history.history["loss"]
    val_loss = model_history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Test loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss (ΜΑΕ)")
    plt.legend()
    plt.show()

visualize_loss(model_history, "Training and Validation Loss")


# In[20]:


plt.figure(figsize=(6,6))
plt.plot(model_history.history['val_r_square'])
plt.plot(model_history.history['r_square'])
plt.title('Model R^2')
plt.ylabel('R^2')
plt.xlabel('Εpochs')
plt.legend(['test','train'],loc='lower right')
plt.show()


# # Πρόβλεψη

# In[21]:


y_pred_train=model.predict(x_train) # Από το train set
y_pred_test=model.predict(x_test)   # Από το test set


# # Συντελεστής συσχέτισης R^2

# In[22]:


r2_train=r2_score(y_train,y_pred_train.reshape(-1,1))
print('From r2_score about training set R^2 =',r2_train)

r2_test=r2_score(y_test,y_pred_test.reshape(-1,1))
print('From r2_score about test set R^2 =    ',r2_test)

print('From model about training set R^2 =',model_history.history['r_square'][-1])

print('From model about test set R^2 =    ',model_history.history['val_r_square'][-1])


# # Στατιστικά

# In[23]:


from sklearn.metrics import mean_absolute_error, mean_squared_error

# Training Set

mae_train=mean_absolute_error(y_train,y_pred_train.reshape(-1,1))
print("Mean absolute error from training set (MAE):      ",mae_train)
mse_train=mean_squared_error(y_train,y_pred_train.reshape(-1,1))
print("Mean squared error from training set (MSE):       ",mse_train)
rmse_train=math.sqrt(mean_squared_error(y_train,y_pred_train.reshape(-1,1)))
print("Root mean squared error from training set (RMSE): ",rmse_train)
bias_train=(y_train-y_pred_train.reshape(-1,1)).mean()
print("Bias from training set:                           ",bias_train)

# Test Set

mae_test=mean_absolute_error(y_test,y_pred_test.reshape(-1,1))
print("\n\nMean absolute error from test set (MAE):      ",mae_test)
mse_test=mean_squared_error(y_test,y_pred_test.reshape(-1,1))
print("Mean squared error from test set (MSE):       ",mse_test)
rmse_test=math.sqrt(mean_squared_error(y_test,y_pred_test.reshape(-1,1)))
print("Root mean squared error from test set (RMSE): ",rmse_test)
bias_test=(y_test-y_pred_test.reshape(-1,1)).mean()  # mesh timi diaforas h diadora meswn timwn???
print("Bias from test set:                           ",bias_test)


# # Αποκανονικοποίηση

# In[24]:


# Training Set
y_train_unNorm=y_train*training_set['PM10'].std()+training_set['PM10'].mean()
y_pred_train_unNorm=y_pred_train*training_set['PM10'].std()+training_set['PM10'].mean()

# Test Set
y_test_unNorm=y_test*test_set['PM10'].std()+test_set['PM10'].mean()
y_pred_test_unNorm=y_pred_test*test_set['PM10'].std()+test_set['PM10'].mean()


# # Comparison Plots

# In[25]:


from sklearn.linear_model import LinearRegression
plt.figure(figsize=(5,5))
plt.title('ΑΝΝ - Comparison Plot for Training Set')
plt.xlim([5,y_train_unNorm.max()+5])
plt.ylim([5,y_train_unNorm.max()+5])

plt.plot(y_train_unNorm,y_pred_train_unNorm.reshape(-1,1),'*',color='blue',label='Data')

plt.plot([y_train_unNorm.min(),y_train_unNorm.max()],[y_train_unNorm.min(),y_train_unNorm.max()],
         color='red',label='Straight line 1-1')

plt.xlabel('PM10')
plt.ylabel('Predicted PM10')
plt.legend()
plt.show()


# In[26]:


plt.figure(figsize=(5,5))
plt.title('ΑΝΝ - Comparison Plot for Test Set')
plt.xlim([15,y_test_unNorm.max()+5])
plt.ylim([15,y_test_unNorm.max()+5])
plt.plot(y_test_unNorm,y_pred_test_unNorm.reshape(-1,1),'*',color='blue',label='Data')

plt.plot([y_test_unNorm.min(),y_test_unNorm.max()],[y_test_unNorm.min(),y_test_unNorm.max()],
         color='red',label='Straight line 1-1')

plt.xlabel('PM10')
plt.ylabel('Predicted PM10')
plt.legend()

plt.show()

