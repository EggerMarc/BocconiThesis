#!/usr/bin/env python
# coding: utf-8

# In[2]:


from keras.layers import Dense, LSTM
from keras.models import Sequential, load_model
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import joblib
#from sklearn.model_selection import tree

from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from collections import deque
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import csv


# In[ ]:


# For simplicity's sake, we'll create a Class to undergo the whole process.
# List of methods to create:
# 1. __init__
# 2. Normalizer
# 3. Train Test split
# 4. Reshaper
# 5. Model creation
# 6. Plotter
# We can do this once we have internet, as we'll have to generalize the parameter split.


# In[ ]:


# Now that we've gotten a succesful model, we can go on to study it a bit in depth.
# 1. We will first push the target minute forward, e.g.: 12:30 input will predict 12:31
# 2. We will add more measures of input, volume, high, low, and close. Note, the close will be of the past minute,
# so if we have 12:30 input, we will put 12:30 close price as well. We can eventually study the accuracy differences
# when this past minute closing price doesn't exist, to increase the trader's trading window
# 3. We will try to implement the VIX, and we will compare the models. Note, the VIX will be an almost daily variable,
# as I'm struggling to find minute trades. 
# 4. At this point we can take metrics, and start the writing of the Thesis


# In[ ]:


# At this point we should normalize the data. We can do it in this code block. I will initially try without
# normalizing, but I suspect the algorithm will yield a straight line.


# In[ ]:


# Note, the volume seems to disrupt predictions at the end of the test period. I suspect it's around the COVID-19
# crisis. We will do two things now.
# First, we will try to predict on longer time stamps, 5M to be specific.
# Second, we will add the VIX, as we suggested before. I'm growing less convinced that the VIX will actually help,
# in fact I think it will disrupt the model, like volume did.


# In[3]:


df_5 = pd.read_csv("EURUSD_M5.csv", delimiter = "\t")
df_1 = pd.read_csv("EURUSD_M1.csv", delimiter = "\t")
# Note, the VIX ends one day before the df_1 and df_5, it'll be on the testing side of the data, so it won't change
# the model
VIX = pd.read_csv("VIX.csv")

for i in range(len(VIX)):
    if VIX['VIXCLS'][i] == '.':
        VIX['VIXCLS'][i] = VIX['VIXCLS'][i-1]

#print(df_1['Time'])
array_1  = []
array_5 = []
for i in range(len(df_1)):
    array_1.append(str.split(df_1['Time'][i]))
    array_5.append(str.split(df_5['Time'][i]))
array_1 = np.array(array_1).T
array_5 = np.array(array_5).T
#df_1.insert(array[0])
df_1['DATE'] = array_1[0]
df_5['DATE'] = array_5[0]

vx_1 = pd.merge(df_1, VIX, on = 'DATE')
vx_5 = pd.merge(df_5, VIX, on = 'DATE')


# In[3]:


# just run this once, we'll export the vix merged data

vx_5.to_csv(r'incl_5.csv', index = False)
vx_1.to_csv(r'incl_1.csv', index = False)

    


# In[4]:


v_open_5, v_high_5, v_low_5, v_close_5, v_volume_5, v_vix_5 = np.array(vx_5['Open']), np.array(vx_5['High']), np.array(vx_5['Low']), np.array(vx_5['Close']), np.array(vx_5['Volume']), np.array(vx_5['VIXCLS'])
v_open_1, v_high_1, v_low_1, v_close_1, v_volume_1, v_vix_1 = np.array(vx_1['Open']),np.array(vx_1['High']),np.array(vx_1['Low']),np.array(vx_1['Close']),np.array(vx_1['Volume']),np.array(vx_1['VIXCLS'])
open_5, high_5, low_5, close_5, volume_5 = np.array(df_5['Open']), np.array(df_5['High']),np.array(df_5['Low']),np.array(df_5['Close']),np.array(df_5['Volume'])
open_1, high_1, low_1, close_1, volume_1 = np.array(df_1['Open']),np.array(df_1['High']),np.array(df_1['Low']),np.array(df_1['Close']),np.array(df_1['Volume'])


# In[16]:


class Trader:
    def __init__(self, pointer_data, actual_data):
        self.pointer_data = pointer_data
        self.actual_data = actual_data
        
    def buy(self, EUR, USD, price):
        USD += EUR/price
        EUR = 0
        return EUR, USD
    
    def sell(self, EUR, USD, price):
        EUR += USD*price
        USD = 0
        return EUR, USD
    
    def tester(self):
        EUR = 100
        USD = 100
        array = []
        for i in range(len(self.actual_data)):
            price = self.actual_data[i]
            if self.pointer_data[i] < self.actual_data[i]:
                USD += EUR*price
                EUR -= EUR
            
            elif 1/self.pointer_data[i] < 1/self.actual_data[i]:
                EUR += USD*(1/price)
                USD -= USD
            array.append(USD + EUR*price)
        #plt.plot(array, color = 'green')

        return array
    
    def summary(self, array):
        x = np.array(array)
        
        print("\nSummary of trading operations: " + " \n\nInitial value: " + str(x[0]) + "\n\nFinal value: " + str(x[-1]))
        print("\nMean gain\loss: " + str(100*(x[-1]-x[0])/x[0]) + "%" + "\n\nStandard deviation: " + str(np.std(x)))
        
        plt.plot(array)
        plt.show()
        
        return
    
        
        
    def _trader(self):
        EUR = 100
        USD = 0
        cash_hist = []
        buys = 0
        sales = 0
        for i in range(len(self.pointer_data)-1):
         #   if pointer_data[i+1] != None:
            price = self.actual_data[i]
            if self.pointer_data[i+1] > self.pointer_data[i]:
                buys += 1
                # Sell
                if buys > 1:
                    EUR, USD = self.sell(EUR, USD, price)                    
                # Buy
                EUR, USD = self.buy(EUR, USD, price)
            elif self.pointer_data[i+1] < self.pointer_data[i]:
                sales += 1
                if sales > 1:
                    EUR, USD = self.buy(EUR, USD, price)
                EUR, USD = self.sell(EUR, USD, price)
                
            cash_hist.append(EUR + USD*price)
        print(sales)
        print(buys)        
        return cash_hist, EUR, USD*price
    
    def _traderWise(self):
        EUR = 100
        USD = 0
        cash_hist = []
        buys = 0
        sales = 0
        for i in range(len(self.pointer_data)-1):
         #   if pointer_data[i+1] != None:
            price = self.actual_data[i]
            if self.pointer_data[i]*0.90 < self.actual_data[i] or self.pointer_data[i]*1.1 > self.actual_data[i]:
                if self.pointer_data[i] > self.actual_data[i]:
                    buys += 1
                    # Sell
                    if buys > 1:
                        EUR, USD = self.sell(EUR, USD, price)                    
                    # Buy
                    EUR, USD = self.buy(EUR, USD, price)
                elif self.pointer_data[i] < self.actual_data[i]:
                    sales += 1
                    if sales > 1:
                        EUR, USD = self.buy(EUR, USD, price)
                    EUR, USD = self.sell(EUR, USD, price)
                
            cash_hist.append(EUR + USD*price)
        print(sales)
        print(buys)        
        return cash_hist, EUR, USD*price


# In[21]:


class my_Models:
# n. 1
# Note, X_data should not be a Pandas Dataframe, but it should already be an np.array of the parameters samples.
# Also, each array within the matrix of X_data should be of n_sample length and only contain one parameter
# Have fun

    def __init__(self, X_data, y_data, _model = None, X_train = None, X_test = None, y_train = None, y_test = None):
        self.X_data = X_data
        self.y_data = y_data
        self._model = _model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
# Just a shortcut to pass through it all in one instance        
    def shortcut(self):
        
        # We do not need to normalize the data. It works just fine
        
        self.X_data = self.Normalizer(self.X_data, self.X_data.shape[0])
        self.y_data = self.Normalizer(self.y_data, 1)
        
        self.X_train, self.X_test, self.y_train, self.y_test = self.splitter()
                
        self.X_train, self.X_test = self.shaper(self.X_train, self.X_test)
            
        return self.X_train, self.X_test, self.y_train, self.y_test

# n. 2
# Note, cols will be the amount of parameters you want in your model to be normalized.
# To ease the process, the parameters that should not be normalized come by the end. E.g.: The first 3 columns are
# normalized, but the last one is left unchecked, cols will be 3.

    def Normalizer(self, data, cols): # This Normalizing method only works well with numpy data. Refrain to Normalizer_2
        # for pandas dataFrames
        
        def column_norm(x):
            scaler = MinMaxScaler()
            x = x.reshape(-1,1)
            scaler.fit(x)
            return scaler.transform(x)
        
        normalized_data = data
        for i in range(cols - 1):
            normalized_data[i] = column_norm(normalized_data[i]).T[0]
        
        return normalized_data
 
    def Normalizer_2(self, data):
        
        #for column in data.columns:
        #    data[column] = MinMaxScaler().fit_transform(data[column])
        
        scaler = MinMaxScaler().fit(data)
        data = scaler.transform(data)
        #scaler_filename = "scaler{}.save".format(str(input()))
        #joblib.dump(scaler, scaler_filename) 
        return data

    def splitter(self, train_len = 0.8):
        train_index = int(train_len*len(self.y_data))
        test_index = int(len(self.y_data) - train_len*len(self.y_data))
        y_train = self.y_data[:train_index]
        y_test = self.y_data[-test_index:]
                
        X_tr = np.zeros([self.X_data.shape[0],train_index])
        X_te = np.zeros([self.X_data.shape[0], test_index])
        
        X_train = self.X_data[:train_index]
        X_test = self.X_data[-test_index:]
        
        X_train = pd.DataFrame(X_train)
        #for i in range(len(self.X_data)):
        #    print(X_tr.shape)
        #    X_tr[i] = self.X_data[i][:train_index]
        #   X_te[i] = self.X_data[i][-test_index:]
        
        return X_train, X_test, y_train, y_test

 # n.3 Will sequence the data  

    def Sequencer(self, data, y_data, sequence_size = 60):
        data = pd.DataFrame(data)
        sequence = deque(maxlen = sequence_size)
        sequenced_data = []
        for i in data.values:
            sequence.append([n for n in i])
            if len(sequence) == sequence_size:
                sequenced_data.append(np.array(sequence))
        
        #self.X_data = np.array(sequenced_data)
        y_data = y_data[sequence_size-1:]
        return np.array(sequenced_data), np.array(y_data)
    
    #def Target()

# n. 4



    # At this point we need the internet to see if we can store data in the class. If not, it'll make the whole class
    # thing a bit slow and frankly useless.
    
# n. 5

    def shaper(self, X_train, X_test):
       # X_train = X_train.T
        X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
        #X_test = X_test.T
        X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1])
        return X_train, X_test

# n. 6    
# Note, at this point the shaping and cleaning of the Data should be done. Here we start doing the model
    
    def RNN(self, X = None):
        try:
            X
        except NameError:
            X = self.X_train
        model = Sequential()
        print(X.shape)
        model.add(LSTM(32, input_shape = (X.shape[1],X.shape[2])))
        
        
        model.add(Dense(10))
        
        #model.add(Dense(5))
        model.add(Dense(1))
        model.compile(loss='mse',optimizer='adam')
        self._model = model
        model.summary()
        return 

# n. 7   
    def Trainer(self, X = None, y = None, x_val = None, y_val = None, export = False, ep = 2, BATCH_SIZE = 30):
        try:
            X
        except NameError:
            X = self.X_train
            y = self.y_train
        
        try:
            x_val
        except NameError:
            x_val = self.x_test
            y_val = self.y_test
            
        
        trained = self._model.fit(X,y,batch_size = BATCH_SIZE,epochs = ep,validation_data = (x_val,y_val))#, steps_per_epoch = 5, validation_steps = 1)
                                  #validation_steps = 10,
                                  #steps_per_epoch = None,
                                 
        
        score = self._model.evaluate(x_val, y_val, verbose=0)
        #print('Test loss:', score[0])
        #print('Test accuracy:', score[1])
        # Save model
        #_model.save("models/{}".format(NAME))
        
        # Here, we'll export the model if needed
        if export == True:
            self._model.save("{}".format(input("Name the file ")))
            print("Succesfully saved")
        
        return trained, self._model
    
    def Plotter(self, x_tst = None, y_tst = None):
        try:
            x_tst
        except NameError:
            x_tst = self.X_test
            y_tst = self.y_test
        predict = self._model.predict(x_tst)
        plt.plot(predict, color = 'orange')
        plt.plot(y_tst)
        plt.ylabel('EUR/USD')
        plt.xlabel('Time')
        plt.legend(['Prediction', 'Actual'], loc='upper left')
        plt.show()
        
    
# n. 8
# Note, here, we will do the second model. The classification algorithm
    def Classify(self, y, _type = 1):
        classified =[]
        if _type == 1:
            for i in range(len(y)):
                try:
                    if y[i+1] > y[i]:
                        classified.append(1)
                    else:
                        classified.append(0)
                except:
                    break
        
        if _type == 2:
            for i in range(len(y)):
                try:
                    change = ((y[i+1] - y[i]) / y[i])
                    if  change > 0.05:
                        print("Type 0")
                        classified.append(0)
                    elif change < 0.05 and change > 0:
                        classified.append(1)
                    elif change < 0 and change < -0.05:
                        print('Type 2')
                        classified.append(2)
                    elif change < -0.05:
                        print('Type 3')
                        classified.append(3)
                except:
                    break           
        return classified

    
    def Random_Forest(self, X, y):

        self._model = RandomForestClassifier(random_state = 1, n_estimators = 100)
        self._model.fit(X,y)
        return self._model
    
    def Classification_NN(self, X, y):
        
        return self._model


# In[28]:



def Sequencer(Data, sequence_size = 60):
    new_data = [] # New Data will be a matrix of shape sample size, sequence size, feature size
    row = []
    Data = np.array(Data)
    ln = len(Data)
    for i in range(ln):
        # For each step of Data, get last sequence_size rows
        row.append(Data[:sequence_size+i])
        try:
            row.append(Data[i+1])
        except:
            print("Done")
        new_data.append(row)
        
    
    return new_data

def Sequencer_2(data, sequence_size = 10):
    sequence = deque(maxlen = sequence_size)
    sequenced_data = []
    print(data)
    for i in data.values:
        sequence.append([n for n in i])
        if len(sequence) == sequence_size:
            sequenced_data.append(sequence)
            
            
    return sequenced_data
    


# In[6]:


vx_1 = vx_1.drop(columns = ['DATE'])
vx_1.set_index('Time',inplace = True)
vx_5 = vx_5.drop(columns = ['DATE'])
vx_5.set_index('Time',inplace = True)

dt_1 = vx_1.drop(columns = ['VIXCLS'])
dt_5 = vx_5.drop(columns = ['VIXCLS'])


# In[7]:


close_1 = vx_1['Close'].shift(1)
close_5 = vx_5['Close'].shift(1)

close_1.fillna(close_1.values[1],inplace=True)
close_5.fillna(close_5.values[1],inplace=True)
#target_1 = Classify(close_1)
#target_5 = Classify(close_5)


# In[9]:


_1 = my_Models(vx_1,close_1)
_5 = my_Models(vx_5,close_5)


# In[10]:


target_1 = _1.Classify(close_1)
target_5 = _5.Classify(close_5)


# In[14]:


# RNN Classified here
def cRNN_1():
    sq = my_Models(dt_1,np.array(target_1))
    X = sq.Normalizer_2(dt_1)
    sq = my_Models(X,target_1)
    x_train, x_test, y_train, y_test = sq.splitter()
    #x_train, x_test = sq.shaper(np.array(x_train),np.array(x_test))
    
    seq_x_train, seq_y_train = sq.Sequencer(x_train,y_train, 60)
    
    seq_x_test, seq_y_test = sq.Sequencer(x_test,y_test, 60)
    
    sq.RNN(seq_x_train)
    trained, model = sq.Trainer(seq_x_train,seq_y_train,seq_x_test,seq_y_test,True)
    #sq.RNN(x_train)
    #trained, model = sq.Trainer(x_train, y_train, x_test, y_test, True)
    return trained, model



def cRNNv_1():
    
    sq = my_Models(vx_1,np.array(target_1))
    X = sq.Normalizer_2(vx_1)
    sq = my_Models(X,target_1)
    x_train, x_test, y_train, y_test = sq.splitter()
    #x_train, x_test = sq.shaper(np.array(x_train),np.array(x_test))
    
    seq_x_train, seq_y_train = sq.Sequencer(x_train,y_train, 60)
    
    seq_x_test, seq_y_test = sq.Sequencer(x_test,y_test, 60)
    
    sq.RNN(seq_x_train)
    trained, model = sq.Trainer(seq_x_train,seq_y_train,seq_x_test,seq_y_test,True)
    return trained, model


    
def cRNN_5():
    sq = my_Models(dt_5,np.array(target_5))
    X = sq.Normalizer_2(dt_5)
    sq = my_Models(X,target_5)
    x_train, x_test, y_train, y_test = sq.splitter()
    #x_train, x_test = sq.shaper(np.array(x_train),np.array(x_test))
    
    seq_x_train, seq_y_train = sq.Sequencer(x_train,y_train, 60)
    
    seq_x_test, seq_y_test = sq.Sequencer(x_test,y_test, 60)
    
    sq.RNN(seq_x_train)
    trained, model = sq.Trainer(seq_x_train,seq_y_train,seq_x_test,seq_y_test,True)
    return trained, model



def cRNNv_5():
    sq = my_Models(vx_5,np.array(target_5))
    X = sq.Normalizer_2(vx_5)
    sq = my_Models(X,target_5)
    x_train, x_test, y_train, y_test = sq.splitter()
    #x_train, x_test = sq.shaper(np.array(x_train),np.array(x_test))
    
    seq_x_train, seq_y_train = sq.Sequencer(x_train,y_train, 60)
    
    seq_x_test, seq_y_test = sq.Sequencer(x_test,y_test, 60)
    
    sq.RNN(seq_x_train)
    trained, model = sq.Trainer(seq_x_train,seq_y_train,seq_x_test,seq_y_test,True)
    return trained, model


# In[ ]:


# RNN here
def RNN_1():
    sq = my_Models(dt_1,np.array(close_1))
    X = sq.Normalizer_2(dt_1)
    sq = my_Models(X,close_1)
    x_train, x_test, y_train, y_test = sq.splitter()
    #x_train, x_test = sq.shaper(np.array(x_train),np.array(x_test))
    
    seq_x_train, seq_y_train = sq.Sequencer(x_train,y_train, 60)
    
    seq_x_test, seq_y_test = sq.Sequencer(x_test,y_test, 60)
    
    sq.RNN(seq_x_train)
    trained, model = sq.Trainer(seq_x_train,seq_y_train,seq_x_test,seq_y_test,True)
    #sq.RNN(x_train)
    #trained, model = sq.Trainer(x_train, y_train, x_test, y_test, True)
    return trained, model



def  RNNv_1():
    
    sq = my_Models(vx_1,np.array(close_1))
    X = sq.Normalizer_2(vx_1)
    sq = my_Models(X,close_1)
    x_train, x_test, y_train, y_test = sq.splitter()
    #x_train, x_test = sq.shaper(np.array(x_train),np.array(x_test))
    
    seq_x_train, seq_y_train = sq.Sequencer(x_train,y_train, 60)
    
    seq_x_test, seq_y_test = sq.Sequencer(x_test,y_test, 60)
    
    sq.RNN(seq_x_train)
    trained, model = sq.Trainer(seq_x_train,seq_y_train,seq_x_test,seq_y_test,True)
    return trained, model


    
def RNN_5():
    sq = my_Models(dt_5,np.array(close_5))
    X = sq.Normalizer_2(dt_5)
    sq = my_Models(X,close_5)
    x_train, x_test, y_train, y_test = sq.splitter()
    #x_train, x_test = sq.shaper(np.array(x_train),np.array(x_test))
    
    seq_x_train, seq_y_train = sq.Sequencer(x_train,y_train, 60)
    
    seq_x_test, seq_y_test = sq.Sequencer(x_test,y_test, 60)
    
    sq.RNN(seq_x_train)
    trained, model = sq.Trainer(seq_x_train,seq_y_train,seq_x_test,seq_y_test,True)
    return trained, model



def RNNv_5():
    sq = my_Models(vx_5,np.array(close_5))
    X = sq.Normalizer_2(vx_5)
    sq = my_Models(X,close_5)
    x_train, x_test, y_train, y_test = sq.splitter()
    #x_train, x_test = sq.shaper(np.array(x_train),np.array(x_test))
    
    seq_x_train, seq_y_train = sq.Sequencer(x_train,y_train, 60)
    
    seq_x_test, seq_y_test = sq.Sequencer(x_test,y_test, 60)
    
    sq.RNN(seq_x_train)
    trained, model = sq.Trainer(seq_x_train,seq_y_train,seq_x_test,seq_y_test,True)
    return trained, model


# In[141]:


# Random Forest here
def RF_1():
    sq = my_Models(dt_1,np.array(target_1))
    X = sq.Normalizer_2(dt_1)
    sq = my_Models(X,np.array(target_1))
    x_train, x_test, y_train, y_test = sq.splitter()
    model = sq.Random_Forest(x_train, y_train)
    result = model.score(x_test,y_test)
    print(result)
    return model



def RFv_1():
    sq = my_Models(vx_1,np.array(target_1))
    X = sq.Normalizer_2(vx_1)
    sq = my_Models(X,np.array(target_1))
    x_train, x_test, y_train, y_test = sq.splitter()
    model = sq.Random_Forest(x_train, y_train)
    result = model.score(x_test,y_test)
    print(result)
    return model


    
def RF_5():
    sq = my_Models(dt_5,np.array(target_5))
    X = sq.Normalizer_2(dt_5)
    sq = my_Models(X,np.array(target_5))
    x_train, x_test, y_train, y_test = sq.splitter()
    model = sq.Random_Forest(x_train, y_train)
    result = model.score(x_test,y_test)
    print(result)
    return model



def RFv_5():
    sq = my_Models(vx_5,np.array(target_5))
    X = sq.Normalizer_2(vx_5)
    sq = my_Models(X,np.array(target_5))
    x_train, x_test, y_train, y_test = sq.splitter()
    model = sq.Random_Forest(x_train, y_train)
    result = model.score(x_test,y_test)
    print(result)
    return model


# In[142]:


test_1 = RF_1()


# In[135]:


rf_model_1 = RF_1()


# In[147]:


joblib.dump(rf_model_v5,"v5_rf")


# In[148]:


joblib.dump(rf_model_1,"rf_1")


# In[149]:


joblib.dump(rf_model_5,"rf_5")


# In[150]:


joblib.dump(rf_model_v1,"rf_v1")


# In[145]:


rf_model_v1 = RFv_1()


# In[137]:


rf_model_5 = RF_5()


# In[146]:


rf_model_v5 = RFv_5()


# In[139]:


joblib.dump(rf_model_1,"rf_1")
joblib.dump(rf_model_v1,"rf_v1")
joblib.dump(rf_model_5,"rf_5")
joblib.dump(rf_model_v5,"rf_v5")


# In[119]:


trained_1, model_1 = RNN_1()


# In[122]:


v_trained_1, v_model_1 = RNNv_1()


# In[134]:


trained_5, model_5 = RNN_5()


# In[126]:


v_trained_5, v_model_5 = RNNv_5()


# In[133]:


sq = my_Models(dt_1,np.array(close_5))
X = sq.Normalizer_2(dt_5)


# In[22]:


c_trained_1, c_model_1 = cRNN_1()


# In[23]:


c_trained_v1, c_model_v1 = cRNNv_1()


# In[24]:


c_trained_5, c_model_5 = cRNN_5()


# In[25]:


c_trained_v5, c_model_v5 = cRNNv_5()


# In[20]:


plt.figure(figsize = (10,10))
fig, axs = plt.subplots(2,2,figsize = (10,10))

fig.suptitle('Training vs Validation Loss')
axs[0,0].plot(c_model_1.history.history['loss'], label = 'Training Loss')
axs[0,0].plot(c_model_1.history.history['val_loss'], label = 'Validation Loss')
#axs[0,0].set_yscale('log')
axs[0,0].grid()
axs[0,0].set(title='1M')
axs[0,0].legend()

axs[0,1].plot(c_model_v1.history.history['loss'])
axs[0,1].plot(c_model_v1.history.history['val_loss'])
axs[0,1].set(title='1M Vix')
#axs[0,1].set_yscale('log')
axs[0,1].grid()

axs[1,0].plot(c_model_5.history.history['loss'])
axs[1,0].plot(c_model_5.history.history['val_loss'])
axs[1,0].set(title='5M')
#axs[1,0].set_yscale('log')
axs[1,0].grid()

axs[1,1].plot(c_model_v5.history.history['loss'], markersize =10)
axs[1,1].plot(c_model_v5.history.history['val_loss'])
axs[1,1].set(title='5M Vix')
axs[1,1].grid()
#axs[1,1].set_yscale('log')

plt.savefig('firstLSTMclassified.png',dpi = 500, bbox_inches = 'tight', pad_inches = 0.1)
plt.show()

