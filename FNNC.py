import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Reshape,  BatchNormalization, concatenate, Input
from keras.layers import Conv2D, MaxPooling2D, SeparableConv2D, ZeroPadding2D, Conv1D, MaxPooling1D
from keras.layers import LSTM, Bidirectional, TimeDistributed, RepeatVector, GRU
from keras.optimizers import Adam, SGD, Nadam
from keras.layers import LeakyReLU
from keras.regularizers import l1, l2
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import scipy.io as sco
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


#随机对原始数据进行采样生成训练集和测试集
def sample(TrainRate, Data, Label):
    n, m = np.shape(Data)
    DataIndex = list(range(m))
    TrainData = []
    TrainLabel = []
    TestData = []
    TestLabel = []
    for i in range(int(TrainRate*m)):
        Randindex = int(np.random.uniform(0, len(DataIndex)))
        TrainData.append(Data[:, Randindex])
        TrainLabel.append(Label[:, Randindex])
        del (DataIndex[Randindex])
    for i in DataIndex:
        TestData.append(Data[:, i])
        TestLabel.append(Label[:, i])
    TrainLabel = np.array(TrainLabel)
    TestLabel = np.array(TestLabel)
    TrainData = np.array(TrainData)
    TestData = np.array(TestData)
    return TrainData, TrainLabel, TestData, TestLabel


def tenfold(Data, Label, iter):
    n, m = np.shape(Data)
    Data = list(np.transpose(Data))
    Label = list(np.transpose(Label))
    m_jr = 10000
    m_ur = m - m_jr
    m_jr = m_jr // 10
    m_ur = m_ur // 10
    TestData = Data[((iter - 1) * m_jr) : (iter * m_jr)] + Data[((iter - 1) * m_ur + m_jr) : (iter * m_ur + m_jr)]
    TestLabel = Label[((iter - 1) * m_jr) : (iter * m_jr)] + Label[((iter - 1) * m_ur + m_jr) : (iter * m_ur + m_jr)]
    TrainData = Data[: ((iter - 1) * m_jr)] + Data[(iter * m_jr):((iter - 1) * m_ur + m_jr)] + Data[(iter * m_ur + m_jr):]
    TrainLabel = Label[: ((iter - 1) * m_jr)] + Data[(iter * m_jr):((iter - 1) * m_ur + m_jr)] + Data[(iter * m_ur + m_jr):]
    TrainLabel = np.array(TrainLabel)
    TrainData = np.array(TrainData)
    TestData = np.array(TestData)
    TestLabel = np.array(TestLabel)
    return TrainData, TrainLabel, TestData, TestLabel


def mix_sample(TrainRate, Data, Label):
    n, m = np.shape(Data)
    m_jr = 10000
    m_ur = m - m_jr
    #对Jasper随机采样
    DataIndex = list(range(m_jr))
    TrainData = []
    TrainLabel = []
    TestData = []
    TestLabel = []
    for i in range(int(TrainRate*m_jr)):
        Randindex = int(np.random.uniform(0, len(DataIndex)))
        TrainData.append(Data[:, Randindex])
        TrainLabel.append(Label[:, Randindex])
        del (DataIndex[Randindex])
    for i in DataIndex:
        TestData.append(Data[:, i])
        TestLabel.append(Label[:, i])
    #对Urban随机采样
    DataIndex = list(range(m_ur))
    for i in range(int(TrainRate * m_ur)):
        Randindex = int(np.random.uniform(0, len(DataIndex)))
        TrainData.append(Data[:, Randindex + m_jr])
        TrainLabel.append(Label[:, Randindex + m_jr])
        del (DataIndex[Randindex])
    for i in DataIndex:
        TestData.append(Data[:, i + m_jr])
        TestLabel.append(Label[:, i + m_jr])
    TrainLabel = np.array(TrainLabel)
    TestLabel = np.array(TestLabel)
    TrainData = np.array(TrainData)
    TestData = np.array(TestData)
    return TrainData, TrainLabel, TestData, TestLabel


def combat_rnn_without_cc(objective, optimizer, metrics ):
    X = Input(shape=(157, 1))

    lstm1 = Bidirectional(LSTM(10, return_sequences=True, dropout=0.2))(X)
    lstm1 = BatchNormalization()(lstm1)
    lstm2 = Bidirectional(LSTM(10, return_sequences=True, dropout=0.3))(lstm1)
    lstm2 = BatchNormalization()(lstm2)
    lstm3 = Bidirectional(LSTM(10, return_sequences=True, dropout=0.4))(lstm2)
    lstm3 = BatchNormalization()(lstm3)
    X2 = Flatten()(lstm3)

    Global = Conv1D(3, 5, activation='relu', padding='same')(X)
    Global = MaxPooling1D(pool_size=2)(Global)

    Global = Conv1D(6, 4, activation='relu')(Global)
    Global = MaxPooling1D(pool_size=2)(Global)

    Global = Conv1D(12, 5, activation='relu')(Global)
    Global = MaxPooling1D(pool_size=2)(Global)

    Global = Conv1D(24, 4, activation='relu')(Global)
    Global = MaxPooling1D(pool_size=2)(Global)

    Global = Flatten()(Global)
    Con = concatenate([X2, Global])

    Den1 = Dense(600, activation='relu', use_bias=None)(Con)
    Global = Dense(150, activation='relu', use_bias=None)(Den1)

    Abadunce = Dense(6, activation='softmax')(Global) 

    model = Model(input=X, output=Abadunce)
    model.compile(loss=objective, optimizer=optimizer, metrics=metrics)
    return model


def combat_rnn_with_cc(objective, optimizer, metrics ):
    X = Input(shape=(157, 1))

    lstm1 = Bidirectional(LSTM(10, return_sequences=True, dropout=0.2))(X)
    lstm1 = BatchNormalization()(lstm1)
    lstm2 = Bidirectional(LSTM(10, return_sequences=True, dropout=0.3))(lstm1)
    lstm2 = BatchNormalization()(lstm2)
    lstm3 = Bidirectional(LSTM(10, return_sequences=True, dropout=0.4))(lstm2)
    lstm3 = BatchNormalization()(lstm3)
    X2 = Flatten()(lstm3)

    Global = Conv1D(3, 5, activation='relu', padding='same')(X)
    Global = MaxPooling1D(pool_size=2)(Global)

    Global = Conv1D(6, 4, activation='relu')(Global)
    Global = MaxPooling1D(pool_size=2)(Global)

    Global = Conv1D(12, 5, activation='relu')(Global)
    Global = MaxPooling1D(pool_size=2)(Global)

    Global = Conv1D(24, 4, activation='relu')(Global)
    Global = MaxPooling1D(pool_size=2)(Global)

    Global = Flatten()(Global)
    Con = concatenate([X2, Global])

    Den1 = Dense(600, activation='relu', use_bias=None)(Con)
    Global = Dense(150, activation='relu', use_bias=None)(Den1)

    Abadunce = Reshape((1, 150))(Global)
    Abadunce = Bidirectional(LSTM(100, dropout=0.2))(Abadunce)
    Abadunce = BatchNormalization()(Abadunce)
    Abadunce = RepeatVector(6)(Abadunce)
    Abadunce = Bidirectional(LSTM(50, return_sequences=True, kernel_regularizer=l2(0.0001), dropout=0.2))(Abadunce)
    Abadunce = BatchNormalization()(Abadunce)
    Abadunce = Bidirectional(LSTM(20, return_sequences=True, kernel_regularizer=l2(0.0001)))(Abadunce)
    Abadunce = TimeDistributed(Dense(1, activation='sigmoid'))(Abadunce)

    model = Model(input=X, output=Abadunce)
    model.compile(loss=objective, optimizer=optimizer, metrics=metrics)
    return model


def pixel_rnn_without_cc(objective, optimizer, metrics):
    X = Input(shape=(157, 1))

    lstm1 = Bidirectional(LSTM(10, return_sequences=True, dropout=0.2))(X)
    lstm1 = BatchNormalization()(lstm1)
    lstm2 = Bidirectional(LSTM(10, return_sequences=True, dropout=0.3))(lstm1)
    lstm2 = BatchNormalization()(lstm2)
    lstm3 = Bidirectional(LSTM(10, return_sequences=True, dropout=0.4))(lstm2)
    lstm3 = BatchNormalization()(lstm3)
    X2 = Flatten()(lstm3)

    Den1 = Dense(600, activation='relu', use_bias=None)(X2)
    Global = Dense(150, activation='relu', use_bias=None)(Den1)
    
    Abadunce = Dense(6, activation='softmax')(Global)

    model = Model(input=X, output=Abadunce)
    model.compile(loss=objective, optimizer=optimizer, metrics=metrics)
    return model


def pixel_rnn_with_cc(objective, optimizer, metrics):
    X = Input(shape=(157, 1))

    lstm1 = Bidirectional(LSTM(10, return_sequences=True, dropout=0.2))(X)
    lstm1 = BatchNormalization()(lstm1)
    lstm2 = Bidirectional(LSTM(10, return_sequences=True, dropout=0.3))(lstm1)
    lstm2 = BatchNormalization()(lstm2)
    lstm3 = Bidirectional(LSTM(10, return_sequences=True, dropout=0.4))(lstm2)
    lstm3 = BatchNormalization()(lstm3)
    X2 = Flatten()(lstm3)

    Den1 = Dense(600, activation='relu', use_bias=None)(X2)
    Global = Dense(150, activation='relu', use_bias=None)(Den1)

    Abadunce = Reshape((1, 150))(Global)
    Abadunce = Bidirectional(LSTM(100, dropout=0.2))(Abadunce)
    Abadunce = BatchNormalization()(Abadunce)
    Abadunce = RepeatVector(6)(Abadunce)
    Abadunce = Bidirectional(LSTM(50, return_sequences=True, kernel_regularizer=l2(0.0001), dropout=0.2))(Abadunce)
    Abadunce = BatchNormalization()(Abadunce)
    Abadunce = Bidirectional(LSTM(20, return_sequences=True, kernel_regularizer=l2(0.0001)))(Abadunce)
    Abadunce = TimeDistributed(Dense(1, activation='sigmoid'))(Abadunce)

    model = Model(input=X, output=Abadunce)
    model.compile(loss=objective, optimizer=optimizer, metrics=metrics)
    return model


def train(lr):
    nadam = Nadam(lr=lr)

    # without chain classfication

    objective = ['categorical_crossentropy']
    metrics = ['cosine_proximity']
    model = pixel_rnn_without_cc(objective, nadam, metrics)

    # with chain classfication

    # objective = ['binary_crossentropy']
    # metrics = ['cosine_proximity']
    # model = pixel_rnn_with_cc(objective, nadam, metrics)

    return model


def pro(TrainData, TestData, TrainLabel, TestLabel):
    m_Train, n_Train = np.shape(TrainData)
    m_Test, n_Test = np.shape(TestData)
    TrainData = TrainData.reshape(m_Train, n_Train, 1)
    TestData = TestData.reshape(m_Test, n_Test, 1)

    # with chain classfication
    # m_Train_L, n_Train_L = np.shape(TrainLabel)
    # m_Test_L, n_Test_L = np.shape(TestLabel)
    # TrainLabel = TrainLabel.reshape(m_Train_L, n_Train_L, 1)
    # TestLabel = TestLabel.reshape(m_Test_L, n_Test_L, 1)

    # without chain classfication
    TrainLabel = TrainLabel
    TestLabel = TestLabel

    return TrainData, TestData, TrainLabel, TestLabel


def eva(Output, Label, metrics):
    if metrics == 'MSE':
       error = Output - Label
       squaredError = []
       absError = []
       for val in error:
           squaredError.append(val * val)  # target-prediction之差平方
           absError.append(abs(val))  # 误差绝对值
       print("MSE = ", sum(squaredError) / len(squaredError))  # 均方误差MSE
    if metrics == 'RMSE':
       from math import sqrt
       error = Output - Label
       squaredError = []
       absError = []
       for val in error:
           squaredError.append(val * val)  # target-prediction之差平方
           absError.append(abs(val))  # 误差绝对值
       MSE = sum(squaredError) / len(squaredError)
       squarMSE = []
       for val in MSE:
            squarMSE.append(sqrt(val))
       RMSE = sum(squarMSE) / len(squarMSE)
       print(squarMSE)
       print("RMSE = ", RMSE) # 均方根误差RMSE


#导入高光谱数据
DataSet = sco.loadmat('Mix/ur.mat')
Data = DataSet.get('Yur')
Label = DataSet.get('Aur')

TrainRate = 0.9
TrainData, TrainLabel, TestData, TestLabel = sample(TrainRate, Data, Label)
TrainData, TestData, TrainLabel, TestLabel = pro(TrainData, TestData, TrainLabel, TestLabel)
lr = 0.0001
model = train(lr)
early_stop = EarlyStopping(monitor='val_loss', mode='auto')
model.fit(TrainData, TrainLabel,  batch_size=1000, epochs=50, validation_split=0.1, callbacks=[early_stop])
model.summary()
prediction = model.predict(TestData, batch_size=1000)
metrics = 'RMSE'
eva(prediction, TestLabel, metrics)