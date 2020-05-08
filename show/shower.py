from base.base_train import BaseTrain
from models.LSTM import LSTM
from tqdm import tqdm
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import seaborn as sns
tf.disable_v2_behavior()


class shower(BaseTrain):
	def __init__(self, sess, model, data, config,logger):
		super(shower, self).__init__(sess, model, data, config, logger)
		self.config = config
		self.model = model
		self.data = data
		self.sess = sess

	def predict(self):
		self.predict=[]
		with tf.variable_scope("lstm",reuse=True):
			pred,_=self.model.predict_build_model()   #预测时只输入[1,time_step,input_size]的测试数据
		prev_seq=self.data.train_x[-1]
		with tf.Session() as sess:
			init = tf.global_variables_initializer()
			sess.run(init)
			self.model.load(sess)
			for i in range(self.config.predict_n):  #预测100个数值
				next_seq=sess.run(pred,feed_dict={self.model.x:[prev_seq]})
				self.predict.append(next_seq[-1])
				#每次得到最后一个时间步的预测结果，与之前的数据加在一起，形成新的测试数据
				prev_seq=np.vstack((prev_seq[1:],next_seq[-1]))
		print(self.predict)

	def predict_2(self):
		test_data = self.data.p_data
		print("in")
		half = int(len(test_data)/2)
		self.half = half
		self.show_pre = []
		config = self.config
		model = self.model
		time_step = self.config.time_step
		print("in")
		for times in range(1,config.predict_n):
			t_data = test_data[times:times+half]
			print(str(times)+"/"+str(half))
			real = t_data[-1:]
			t_data = t_data[0:-1]
			normalize_t_data=(t_data-np.mean(t_data))/np.std(t_data)  #对数据进行标准化 （数据 - 均值）/（方差）
			normalize_t_data=normalize_t_data[:,np.newaxis]       #增加数据的维度，使数据维度相同
			train_x = self.data.train_x
			train_y = self.data.train_y

			for i in range(len(normalize_t_data)-time_step-1):
				x=normalize_t_data[i:i+time_step]
				y=normalize_t_data[i+1:i+time_step+1]
				train_x.append(x.tolist())
				train_y.append(y.tolist()) 
			with tf.variable_scope('lstm',reuse=True):
				pred,last= model.predict_build_model()
			with tf.Session() as sess:
				#参数恢复
				init = tf.global_variables_initializer()
				sess.run(init)
				model.load(sess)
				#I run the code in windows 10,so use  'model_save1\\modle.ckpt'
				#if you run it in Linux,please use  'model_save1/modle.ckpt'

				#取训练集最后一行为测试样本。shape = [1,time_step,input_size]
				prev_seq=train_x[-1]
				print(prev_seq)
				predict=[]
				#得到之后的n个预测结果
				for i in range(2):  #预测n个数值
					next_seq=sess.run(pred,feed_dict={model.x:[prev_seq]})
					#print(next_seq)
					predict.append(next_seq[-1])
					#每次得到最后一个时间步的预测结果，与之前的数据加在一起，形成新的测试数据
					prev_seq=np.vstack((prev_seq[1:],next_seq[-1]))       
				predict = np.squeeze(predict)#减少数据维度
				predict = predict*np.std(t_data)+np.mean(t_data)#恢复先前价格
				print("实际："+str(real[0]))
				print("预测: "+str(predict[-1]))
				print("起价: "+str(t_data[-1])) 
				self.show_pre.append(predict[0])

	def show(self):
		predict = self.predict
		data = self.data.t_data
		print(np.mean(data))
		print(np.std(data))
		#以折线图展示结果
		plt.figure(figsize = (20,20)) #图像大小为8*8英寸
		#设置背景风格
		sns.set_style(style = 'whitegrid') #详细参数看seaborn的API  http://seaborn.pydata.org/api.html
		#设置字体
		sns.set_context(context = 'poster',font_scale = 1)
		plt.title('Prediction')
		predict = np.squeeze(predict)#减少数据维度
		predict = predict*np.std(data)+np.mean(data)#恢复先前价格
		predict = np.insert(predict,0,data[-1])
		#print(data[-1])
		#print(predict)
		plt.plot(list(range(len(data))), data, color='b',label = 'raw data') #这是原来股票的价格走势，用蓝色曲线表示
		plt.plot(list(range(len(data)-1, len(data) + len(predict)-1)), predict, color='r',label = 'predict trend') #预测未来的价格走势用红色表示
		plt.legend(loc = 'best')
		plt.xlabel('Time')
		plt.ylabel('Price')
		plt.show()

	def show2(self):
		show_pre = self.show_pre
		data = self.data.t_data
		half = self.half
		test_data = self.data.p_data
		n= self.config.predict_n
		print(np.mean(data))
		print(np.std(data))
		#以折线图展示结果
		plt.figure(figsize = (20,20)) #图像大小为8*8英寸
		#设置背景风格
		sns.set_style(style = 'whitegrid') #详细参数看seaborn的API  http://seaborn.pydata.org/api.html
		#设置字体
		sns.set_context(context = 'poster',font_scale = 1)
		plt.title('Prediction')
		plt.plot(list(range(half+n)), test_data[0:half+n], color='b',label = 'raw data') #这是原来股票的价格走势，用蓝色曲线表示
		plt.plot(list(range(half-4,half + len(show_pre)-4)), show_pre, color='r',label = 'predict trend') #预测未来的价格走势用红色表示
		plt.legend(loc = 'best')
		plt.xlabel('Time')
		plt.ylabel('Price')
		plt.show()
	def show3(self):
		show_pre = self.show_pre
		data = self.data.t_data
		half = self.half
		test_data = self.data.p_data
		n= self.config.predict_n
		print(np.mean(data))
		print(np.std(data))
		#以折线图展示结果
		plt.figure(figsize = (20,20)) #图像大小为8*8英寸
		#设置背景风格
		sns.set_style(style = 'whitegrid') #详细参数看seaborn的API  http://seaborn.pydata.org/api.html
		#设置字体
		sns.set_context(context = 'poster',font_scale = 1)
		plt.title('Prediction')
		plt.plot(list(range(4+n)), test_data[half-4:half+n], color='b',label = 'raw data')  #这是原来股票的价格走势，用蓝色曲线表示
		plt.plot(list(range(1,1+len(show_pre))), show_pre, color='r',label = 'predict trend') #预测未来的价格走势用红色表示
		plt.legend(loc = 'best')
		plt.xlabel('Time')
		plt.ylabel('Price')
		plt.show()

