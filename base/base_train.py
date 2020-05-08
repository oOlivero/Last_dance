import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
class BaseTrain:
    def __init__(self, sess, model, data, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.data = data
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

    def train(self):
        loss_list = []
        flag = 0
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess)+2, self.config.num_epochs, 1):
            loss_list.extend(self.train_epoch())
            self.sess.run(self.model.increment_cur_epoch_tensor)
            print("eproch:"+str(cur_epoch))
            self.sess.run(self.model.global_step_tensor)
            flag =1
        if flag==1:
            print("The train has finished")
            print(len(loss_list))
            sns.set_style(style = 'whitegrid') #详细参数看seaborn的API  http://seaborn.pydata.org/api.html
            #设置字体
            sns.set_context(context = 'poster',font_scale = 1)
            plt.figure(figsize = (20,20)) #图像大小为20*20英寸
            plt.plot(np.arange(0,len(loss_list)),loss_list,'+-',color = 'g')
            plt.title('Loss trend')
            plt.ylabel('Loss')

    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError