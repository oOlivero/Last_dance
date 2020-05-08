import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import datetime
import time
import yfinance as yf
import json
from pprint import pprint
class DataGenerator:
	def __init__(self, config):
		self.p_data = self.get_stock_data(config.p_stock,config.p_start,config.p_end)
		self.t_data = self.get_stock_data(config.t_stock,config.t_start,config.t_end)
		self.config = config
		self.get_train_data();
		# load data here
	def get_stock_data(self,name,start,end):
		try:
			yf.pdr_override()
			#实时股票数据
			if type(start) is str:
				start = datetime.datetime.strptime(start,'%Y/%m/%d')
			if type(end) is str:
				end = datetime.datetime.strptime(end,'%Y/%m/%d')
			finance = pdr.get_data_yahoo(name,start,end) 
			data = np.array(finance['Close']) #获取收盘价的数据
			data = data[::1] #获取这列的所有数据
			print('股票数据获取完成！！')
			return data
		except Exception:
			print('股票数据获取失败！！')
	def get_train_data(self):
		#print(np.mean(self.t_data))
		#print(np.std(self.t_data))
		train_x,train_y=[],[]   #训练集
		normalize_data=(self.t_data-np.mean(self.t_data))/np.std(self.t_data)  #对数据进行标准化 （数据 - 均值）/（方差）
		normalize_data=normalize_data[:,np.newaxis]       #增加数据的维度，使数据维度相同
		time_step = self.config.time_step
		for i in range(len(normalize_data)-time_step-1):
			x=normalize_data[i:i+time_step]
			y=normalize_data[i+1:i+time_step+1]
			train_x.append(x.tolist())
			train_y.append(y.tolist()) 
		self.train_x = train_x
		self.train_y = train_y
		#print(train_x)
		#print(train_y)