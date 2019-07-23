import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt

HIDDEN_SIZE = 30
NUM_LAYERS = 2

TIMESTEPS = 10
TRAINING_STEPS = 10000
BATCH_SIZE = 32

TRAINING_EXAMPLES = 10000
TESTING_EXAMPLES = 1000
SAMPLE_GAP = 0.01

def generate_data(seq):
    X = []
    y = []
    # 序列的第i项和后面的TIMESTEPS - 1项合在一起作为输入；第i + TIMESTEPS项作为输
    # 出。即用sin函数前面的TIMESTEPS个点的信息,预测第i + TIMESTEPS    个点的函数值。
    for i in range(len(seq) - TIMESTEPS):
        X.append([seq[i: i + TIMESTEPS]])
        y.append([seq[i + TIMESTEPS]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def lstm_model(X, y , is_training):
    # 使用多层的LSTM 结构。
    cell= tf.nn.rnn_cell.MultiRNNCell([
        tf. nn.rnn_cell.BasicLSTMCell (HIDDEN_SIZE)
        for _ in range (NUM_LAYERS)])

    # 使用TensorFlow 接口将多层的LSTM 结构连接成RNN 网络并计算其前向传播结果。
    outputs, _= tf.nn.dynamic_rnn(cell , X, dtype=tf.float32)
    # outputs ；在顶层LSTM 在每一步的输出结果,它的维度是［ batch size , time ,
    # HIDDEN SIZE ］ 。在本问题中只关注最后一个时刻的输出结果。
    output = outputs [ :, - 1 , : ]
    # 对LSTM 网络的输出再做hll 一层仓链接层并计算损失。注意这里默认的损失为平均
    # 平方差损失函数。
    predictions = tf.contrib.layers.fully_connected(
            output , 1 , activation_fn=None)

    # 只在训练时计算损失扇数和优化步骤。测试时直接返回预测结果。
    if not is_training :
        return predictions , None , None

    # 计算损失函数。
    loss = tf.losses. mean_squared_error ( labels=y, predictions=predictions)
    # 创建模型优化器并得到优化步骤。
    train_op= tf.contrib.layers.optimize_loss(
            loss , tf. train.get_global_step(),
            optimizer= "Adagrad", learning_rate=0.1)
    return predictions , loss , train_op

def train(sess , train_X , train_y) :
    # 将训练数据以数据集的方式提供给计算图。
    ds = tf.data.Dataset.from_tensor_slices((train_X, train_y))
    ds = ds.repeat() .shuffle(1000) .batch(BATCH_SIZE)
    X, y = ds.make_one_shot_iterator().get_next()
    # 调用模型, 得到预测结果、损失丽数,和训练操作。
    with tf.variable_scope ( "model"):
        predictions , loss , train_op = lstm_model(X , y , True)
        # 初始化变量。
        sess.run ( tf. global_variables_initializer() )
        for i in range (TRAINING_STEPS) :
            _, l = sess.run([train_op , loss])
            if i %  100 == 0 :
                print (" train step:" + str(i)+ "loss :" + str(l))


def run_eval(sess , test_X, test_y) :
    # 将测试数据以数据集的方式提供给计算圈。
    ds = tf.data.Dataset.from_tensor_slices((test_X, test_y))
    ds = ds.batch(1)
    X, y = ds.make_one_shot_iterator().get_next()
    # 调用模型得到计算结果。这里不需要输入真实的y 值
    with tf.variable_scope ("model", reuse=True) :
        prediction , _,_ = lstm_model (X , [0.0], False)

    # 将预测结果在入一个数组。
    predictions = []
    labels = []
    for i in range (TESTING_EXAMPLES):
        p , l = sess.run([prediction , y])
        predictions.append(p)
        labels.append(l)

    # 计算rmse 作为评价指标。
    predictions= np.array(predictions).squeeze()
    labels = np.array(labels).squeeze()
    rmse = np.sqrt(((predictions - labels) ** 2) .mean(axis=0))
    print (" Mean Square Error is ： %g" % rmse)

    # 对预测的sin 函数曲线进行绘图,得到的结果如图8 - 12 所示。
    plt.figure()
    plt.plot(predictions , label = 'predictions')
    plt.plot(labels , label = 'real_sin')
    plt.legend()
    plt. show()

# 用正弦函数也成训练和测试数据集合。
# numpy. linspace 函数可以创建一个等差序列的数组, 它常用的参数有三个参数,第一个参数
# 表示起始值,第二个参数次示终止值,第三个参数表示数列的长度。例如, linspace (1, 10 ,10)
# 产生的数组是arrray ( [1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 ］） 。
test_start= (TRAINING_EXAMPLES+ TIMESTEPS)* SAMPLE_GAP
test_end= test_start+(TESTING_EXAMPLES + TIMESTEPS ) * SAMPLE_GAP
train_X, train_y = generate_data(np.sin (np .linspace (
    0 , test_start , TRAINING_EXAMPLES + TIMESTEPS , dtype=np.float32)))

test_X, test_y = generate_data(np.sin(np.linspace(
    test_start , test_end , TESTING_EXAMPLES+ TIMESTEPS, dtype=np. float32)))

with tf.Session() as sess:
    # 训练棋型。
    train(sess , train_X , train_y)
    # 使用训练好的模型对测试数据进行预测。
    run_eval(sess , test_X , test_y)