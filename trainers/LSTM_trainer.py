from base.base_train import BaseTrain
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class LSTM_trainer(BaseTrain):
    def __init__(self, sess, model, data, config,logger):
        super(LSTM_trainer, self).__init__(sess, model, data, config,logger)
        self.index = 1
    def train_epoch(self):
        train_x = self.data.train_x
        train_y = self.data.train_y
        losses = []
        accs = []
        start=0
        batch_size = self.config.batch_size
        end=start+batch_size
        step = 0
        while(end<len(train_x)):
            feed_dict={self.model.x:train_x[start:end],self.model.y:train_y[start:end]}
            _,loss=self.sess.run([self.model.train_step,self.model.cross_entropy],feed_dict = feed_dict)
            start+=batch_size
            end=start+batch_size
            losses.append(loss)
            if step%10==0:
                print("Number of iterations:",self.index," loss:",loss)
            step+=1

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
        'loss': loss,
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)
        self.index += 1
        return losses