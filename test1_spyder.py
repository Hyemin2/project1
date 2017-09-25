# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 20:56:26 2017

@author: Hyemin
"""

from keras.models import Model
from keras.layers import Input, Dense, Lambda, Activation
from keras.layers import add, multiply, concatenate
from keras.layers import Dropout
from keras import backend as K
import numpy as np
import pandas as pd
import pandas as Series
import pandas as DataFrame
from pandas import concat
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM 
from keras.layers import GRU
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from keras.layers import Activation
from keras.layers.advanced_activations import LeakyReLU
from matplotlib import pyplot
from keras.callbacks import History 

df=pd.read_csv(r'C:\Users\Hyemin\feature_4777_9_19_9_21_train.csv')
df_test=pd.read_csv(r'C:\Users\Hyemin\feature_4777_9_22_test.csv')

df.head()

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var{0}(t-{1})'.format(j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names +=[('var{0}(t)'.format(j+1)) for j in range(n_vars)]
		else:
			names +=[('var{0}(t+{1})'.format(j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

past=10
future=4777
feature=5
a=series_to_supervised(df, past,future)
a_test=series_to_supervised(df_test, past,future)
#print(a)
#print(a_test)

train=a.values  #X와 Y로 나누려면 꼭 PANDAS DATAFRAME에서 NUMPY로 바꿔야 한다
print(np.shape(train))

test=a_test.values
#print(np.shape(test))

"""데이터 [0,1] 사이로 변환"""
scaler=MinMaxScaler(feature_range=(0,1))
scaler=scaler.fit(train)
scaler=scaler.fit(test)
normalized_train=scaler.transform(train)
normalized_test=scaler.transform(test)
#print(np.shape(normalized_train))
#print(np.shape(normalized_test))

def x_y_data(data,a): #a는 미래 예측할 step수:4 이다
    x=list()
    y=list()
    for i in range(len(data)):
        in_x=data[i, :-a]
        in_y=data[i, -a:]
        x.append(in_x)
        y.append(in_y)
        
    x=np.array(x)
    y=np.array(y)
    return x, y

"""트레이닝 데이터 데이터 x와 y 분해"""
b=feature*future
train_x, train_y=x_y_data(normalized_train,b) #4777*(미래 예측할 수)=9554
#print('train_x', np.shape(train_x))
#print('train_y',np.shape(train_y))

"""테스트 데이터 데이터 x와 y 분해"""
test_x, test_y=x_y_data(normalized_test,b) #4777*(미래 예측할 수)=9554
#print('test_x', np.shape(test_x))
#print('test_y',np.shape(test_y))

"""트레이닝 x와 y 3차원으로 reshape"""
train_x=train_x.reshape(len(train_x),past,feature)  
train_y=train_y.reshape(len(train_y),future,feature) 

print('train_x', np.shape(train_x))
print('train_y',np.shape(train_y))

"""테스트 x와 y 3차원으로 reshape"""
test_x=test_x.reshape(len(test_x),past,feature)  
test_y=test_y.reshape(len(test_y),future,feature) 

print('test_x', np.shape(test_x))
print('test_y',np.shape(test_y))

