from base.base_model import BaseModel
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class LSTM(BaseModel):
    def __init__(self, config):
        super(LSTM, self).__init__(config)
        self.init_data()
        with tf.variable_scope("lstm"):
            self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        # network architecture
        rnn_unit = self.config.rnn_unit
        weights = self.weights
        biases = self.biases
        w_in=weights['in']
        b_in=biases['in']
        input=tf.reshape(self.x,[-1,self.config.input_size])  #需要将tensor转为2维进行计算，计算后的结果作为 隐藏层的输入
        input_rnn=tf.matmul(input,w_in)+b_in
        input_rnn=tf.reshape(input_rnn,[-1,self.config.time_step,rnn_unit])   #将tensor转为3维，作为 lstm cell的输入
        cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
        init_state=cell.zero_state(self.config.batch_size,dtype=tf.float32)
        output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)
        output=tf.reshape(output_rnn,[-1,rnn_unit])  #作为输出层的输入
        w_out=weights['out']
        b_out=biases['out']
        pred=tf.matmul(output,w_out)+b_out

        self.cross_entropy = tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(self.y, [-1])))
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.train_step = tf.train.AdamOptimizer(self.config.lr).minimize(self.cross_entropy)


    def predict_build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        # network architecture
        rnn_unit = self.config.rnn_unit
        weights = self.weights
        biases = self.biases
        w_in=weights['in']
        b_in=biases['in']
        input=tf.reshape(self.x,[-1,self.config.input_size])  #需要将tensor转为2维进行计算，计算后的结果作为 隐藏层的输入
        input_rnn=tf.matmul(input,w_in)+b_in
        input_rnn=tf.reshape(input_rnn,[-1,self.config.time_step,rnn_unit])   #将tensor转为3维，作为 lstm cell的输入
        cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
        init_state=cell.zero_state(1,dtype=tf.float32)#预测时只输入[1,time_step,input_size]的测试数据
        output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)
        output=tf.reshape(output_rnn,[-1,rnn_unit])  #作为输出层的输入
        w_out=weights['out']
        b_out=biases['out']
        pred=tf.matmul(output,w_out)+b_out
        return pred,final_states
    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def init_data(self):
        rnn_unit = self.config.rnn_unit
        self.x = tf.placeholder(tf.float32, [None,self.config.time_step,self.config.input_size])
        self.y = tf.placeholder(tf.float32, [None,self.config.time_step,self.config.output_size])
        self.weights={
        'in':tf.Variable(tf.random_normal([self.config.input_size,self.config.rnn_unit])),
        'out':tf.Variable(tf.random_normal([rnn_unit,1]))
        }
        self.biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
        }
