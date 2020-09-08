
from keras.layers import Dense, LSTM
from keras.models import Sequential
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import csv
from collections import deque

epoca = 1

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

        self.X_train = self.shaper(self.X_train)
        self.X_test = self.shaper(self.X_test)

        return self.X_train, self.X_test, self.y_train, self.y_test

# n. 2
# Note, cols will be the amount of parameters you want in your model to be normalized.
# To ease the process, the parameters that should not be normalized come by the end. E.g.: The first 3 columns are
# normalized, but the last one is left unchecked, cols will be 3.

    def Normalizer(self, data):

        data = MinMaxScaler().fit_transform(data)
        data = pd.DataFrame(data)

        return data

# n. 3

    def splitter(self, train_len = 0.8):
        train_index = int(train_len*len(self.y_data))
        test_index = int(len(self.y_data) - train_len*len(self.y_data))
        y_train = self.y_data[:train_index]
        y_test = self.y_data[-test_index:]

        X_train = self.X_data[:train_index]
        X_test = self.X_data[-test_index:]

        return X_tr, X_te, y_train, y_test

    # At this point we need the internet to see if we can store data in the class. If not, it'll make the whole class
    # thing a bit slow and frankly useless.

    def Sequencer(self, data, y_data, sequence_size = 5):
        sequence = deque(maxlen = sequence_size)
        sequenced_data = []
        for i in data.values:
            sequence.append([n for n in i])
            if len(sequence) == sequence_size:
                sequenced_data.append(np.array(sequence))

        #self.X_data = np.array(sequenced_data)
        y_data = y_data[sequence_size-1:]
        return np.array(sequenced_data), np.array(y_data)
# n. 4

    def shaper(self, X_train):
        X_train = X_train.T
        X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
        return X_train

# n. 5
# Note, at this point the shaping and cleaning of the Data should be done. Here we start doing the model

    def RNN(self, X = None):
        if X == None:
            X = self.X_train
        model = Sequential()
        model.add(LSTM(32, input_shape = (X.shape[1],X.shape[2])))
        model.add(Dense(10))
        #model.add(Dense(50))
        model.add(Dense(1))
        model.compile(loss='mse',optimizer='adam')
        self._model = model
        model.summary()
        return

# n. 6
    def Trainer(self, X = None, y = None, x_val = None, y_val = None, export = False, ep = 30, BATCH_SIZE = 30):
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

    def plotter(self):
        predict = self._model.predict(self.X_test)
        plt.plot(predict, color = 'orange')
        plt.plot(self.y_test)
        plt.ylabel('EUR/USD')
        plt.xlabel('Time')
        plt.legend(['Prediction', 'Actual'], loc='upper left')
        plt.show()
# n. 7
# Note, here, we will do the second model. The classification algorithm
    def Classify(self, y, _type = 1):
        classified =[]
        if _type == 1:
            for i in range(len(y_train)):
                try:
                    if y_train[i+1] > y_train[i]:
                        classified.append(1)
                    else:
                        classified.append(0)
                except:
                    break

        if _type == 2:
            for i in range(len(y_train)):
                try:
                    change ((y_train[i+1] - y_train[i]) / y_train[i])
                    if  change > 0.05:
                        classified.append(0)
                    elif change < 0.05 and change > 0:
                        classified.append(1)
                    elif change < 0 and change < -0.05:
                        classified.append(2)
                    elif change < -0.05:
                        classified.append(3)
                except:
                    break
        return classified

    def Random_Forest(self, X, y):

        self._model = RandomForestClassifier(random_state = 1)
        self._model.fit(X,y)
        return self._model


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
