
import pandas as pd
import numpy as np
import matplotlib as plt
import os, time, sys
from algo import *
from strategos import *
#import data.py
from keras.models import load_model
from data import *



model_1 = load_model('model_1')
model_5 = load_model('model_5')
model_v1 = load_model('v_model_1')
model_v5 = load_model('v_model_5')

def catch():
    V1 = pd.read_csv('trade_M1.csv',delimiter='\t')
    V5 = pd.read_csv('trade_M5.csv',delimiter='\t')

#    V1 = pd.read_csv('EURUSD_M1.csv',delimiter='\t')
#    V5 = pd.read_csv('EURUSD_M5.csv',delimiter='\t')
    return V1, V5

V1, V5 = catch()
print(V1.head)
print(V5.head)

array_1  = []
array_5 = []

for i in range(len(VIX)):
    if VIX['VIXCLS'][i] == '.':
        VIX['VIXCLS'][i] = VIX['VIXCLS'][i-1]


for i in range(len(V1)):
    array_1.append(str.split(V1['Time'].values[i]))

for n in range(len(V5)):
    array_5.append(str.split(V5['Time'].values[i]))


array_1 = np.array(array_1).T
array_5 = np.array(array_5).T
#df_1.insert(array[0])
V1['DATE'] = array_1[0]
V5['DATE'] = array_5[0]
# v_data_5, v_data_1, data_5, data_1
# close_5, close_1
def _main_(close, data, model):

    tv5 = my_Models(data, close)
    v5 = tv5.Normalizer(data,5)
    c5 = tv5.Normalizer(close,1)

    v5 = tv5.shaper(v5)

    v5_st = strategy(model)
    v5_st.function(v5, close)
    return v5_st

t1 = _main_(close_1,data_1,model_1)
t1_ = _main_(close_1,v_data_1,model_v1)
t5 = _main_(close_5,data_5,model_5)
t5_ = _main_(close_5,v_data_5,model_v5)

#print("Comprehensive return summary: \nVix 1M : {}\nNo Vix 1M : {}\nVix 5M : {}\nNo Vix 5M : {}".format(round(100*((t1_[-1] - 1000)/ 1000)),round(100*((t1[-1] - 1000)/ 1000)),round(100*((t5_[-1] - 1000)/ 1000)),round(100*((t5[-1] - 1000)/ 1000))))

t1.plotter()
t1_.plotter()
t5.plotter()
t5_.plotter()
