import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class strategy:
    def __init__(self, model, USD = [0], EUR = [0], holding = [1000], predictions = None, final = []):
        self.model = model
        self.USD = USD
        self.EUR = EUR
        self.holding = holding
        self.predictions = predictions
        self.final = final


    def buy(self, price):
        self.EUR.append(self.holding[-1]/price)

        return

    def sell(self, price):
        self.USD.append(self.holding[-1]*price)

        return

    def close(self, price):
        if self.USD[-1] != 0 or self.EUR[-1] != 0:
            self.holding.append(self.USD[-1]/price + self.EUR[-1]*price)
            self.USD.append(0)
            self.EUR.append(0)
        return

    def function_2(self, X_data, y_data):
        empty = []
        self.holding = [1000] # Euros
        USD = []
        self.predictions = self.model.predict(X_data)
        for n in range(len(X_data)):
            # The idea is to start off with 1000 EUR. If the price is expected to go up, we hold the
            # euros. If the price is expected to go down, we switch to USD.

            # Hold the euro
            if self.predictions[n] > y_data[n]:
                # Keep Euros instead of USD
                #self.buy(y_data[n])
                # If we previously owned USD, we will want to convert them into EUR
                try:
                    if USD[n-1] != 0:
                        self.holding.append(USD[n-1]/y_data[n])
                        USD.append(0)
                    else:
                        self.holding.append(self.holding[n-1])
                except IndexError:
                    pass
            # If USD appreciates in value
            else:
                # We want to check if we hold USD, if so, we hold. Else, we buy USD
                try:
                    USD[n-1]
                    if USD[n-1] != 0:
                        USD.append(USD[n-1])
                    else:
                        USD.append(self.holding[n-1]*y_data[n])
                        self.holding.append(0)
                except IndexError:
                    pass

            empty = []
            print("{}% of the simulation complete".format(round(n/len(X_data)*100)), end='\r')
        print(self.holding)
        print(USD)
        plt.figure(figsize = (10,10))
        plt.plot(self.holding)
        plt.grid()
        plt.show()
        return

    def plotter(self):

    #    plt.plot(self.holding)
    #    plt.show()
        return self.final, self.predictions

    def function(self, X_data, y_data):
        empty = []
        self.holding = [1000]
        self.predictions = []
        self.predictions = self.model.predict(X_data)
        for n in range(len(X_data)):
            #empty.append(X_data[n-5])
            #prediction = self.model.predict(np.array(empty))
            self.close(y_data[n])
            if n <10:
                print(y_data[n])
            #self.predictions.append(prediction)
            if self.predictions[n] > y_data[n]:
                # Keep Euros instead of USD
                self.buy(y_data[n])
            if self.predictions[n] < y_data[n]:
                self.sell(y_data[n])
            empty = []
            print("{}% of the simulation complete".format(round(n/len(X_data)*100)), end='\r')
        plt.figure(figsize = (10,10))
        plt.plot(self.holding)
        plt.grid()
        plt.show()
        return self.holding

    def function_classified(self, X_data, y_data):
        self.holding = [1000]
        self.predictions = []
        self.predictions = self.model.predict(X_data)
        for n in range(len(X_data)):
            #empty.append(X_data[n-5])
            #prediction = self.model.predict(np.array(empty))
            self.close(y_data[n])
            #self.predictions.append(prediction)
            if np.round(self.predictions[n]) == 1:
                # Keep Euros instead of USD
                self.buy(y_data[n])
            if np.round(self.predictions[n]) == 0:
                self.sell(y_data[n])
            print("{}% of the simulation complete".format(round(n/len(X_data)*100)), end='\r')
        plt.figure(figsize = (10,10))
        plt.plot(self.holding)
        plt.grid()
        plt.show()
        self.EUR = [0]
        self.USD = [0]
        return self.holding
