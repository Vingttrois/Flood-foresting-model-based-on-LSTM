# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 14:10:31 2023

@author: FanW
"""

from keras import backend as K
from keras.layers import LSTM, RepeatVector, Dense, Dot, Input,\
     Activation, Add, Reshape, Lambda, Multiply, Concatenate, Layer
from keras.models import Model



#双阶段注意力Encoder-Decoder架构
class Transpose_Tensor(Layer): #张量转置封装为Layer
    def __init__(self, axis, **kwargs):
        self.axis = axis
        super(Transpose_Tensor, self).__init__(**kwargs)
    def build(self, input_shape):
        super(Transpose_Tensor, self).build(input_shape)
    def call(self, inputs, **kwargs):
        return K.permute_dimensions(inputs, pattern=self.axis)
    def compute_output_shape(self, input_shape):
        return (input_shape[self.axis[0]], input_shape[self.axis[1]], input_shape[self.axis[2]])
    
def Encoder_Attention_one_step(X, h_prev, s_prev):
    N = K.int_shape(X)[1] #输入序列长度
    n_K_en = K.int_shape(X)[2] #输出变量个数
    #[ht-1,st-1]
    concat = Concatenate()([h_prev, s_prev])  #(None,2n), n为Encoder隐藏层大小
    #[ht-1,st-1]We, 不使用激活函数和偏置项, We大小为（2n*N）
    temp1 = Dense(N, use_bias=False)(concat)   #(None,N)
    temp2 = RepeatVector(n_K_en,)(temp1)  #(None,n_K_en,N)
    X_temp = Transpose_Tensor((0, 2, 1))(X) #(None,n_K_en,N) 
    #XUe, Ue大小为（N*N）
    temp3 = Dense(N, use_bias=False)(X_temp)  #(None,n_K_en,N)
    #[ht-1,st-1]We + XUe
    temp4 = Add()([temp2, temp3])  #(None,n_K_en,N)
    #tanh([ht-1,st-1]We + XUe)
    temp5 = Activation(activation='tanh')(temp4)  #(None,n_K_en,N)
    #(tanh([ht-1,st-1]We + XUe))Ve, Ve大小（N*1)
    temp6 = Dense(1, use_bias=False)(temp5) #(None,n_K_en,1)
    temp7 = Transpose_Tensor((0, 2, 1))(temp6) #(None,1,n_K_en)
    #softmax(eik)
    alphas = Activation(activation='softmax')(temp7) #(None,1,n_K_en)
    return alphas


def Encoder_Attention(X, h0_en, s0_en):
    N = K.int_shape(X)[1] #输入序列长度
    n_K_en = K.int_shape(X)[2] #输入变量个数
    n = K.int_shape(h0_en)[1] #Encoder隐藏单元个数
    
    attention_weight_t = None 
    h = h0_en
    s = s0_en
    #Encoder Attention部分
    for i in range(N):
        #计算权重
        context = Encoder_Attention_one_step(X, h, s) #(None,1,n_K_en)
        if i != 0:
            attention_weight_t = Concatenate()([attention_weight_t, context])
        else:
            attention_weight_t = context
        x_temp = Lambda(lambda x: X[:, i, :])(X) #(None,n_K_en)
        x = Reshape((1, n_K_en))(x_temp) #(None,1,n_K_en)
        h, _, s = LSTM(n, return_state=True) (x, initial_state=[h, s]) #(None, n)
        
    attention_weight_t = Reshape((N, n_K_en))(attention_weight_t) #(None,N,n_K_en)
    #加权
    X_ = Multiply()([attention_weight_t, X])   #(None,N,n_K_en)
    return X_


def Decoder_Attention_one_step(h_de_prev, s_de_prev, h_en_all):
    n = K.int_shape(h_en_all)[2]
    N = K.int_shape(h_en_all)[1]
    #[ht-1, st-1]
    concat = Concatenate()([h_de_prev, s_de_prev])  #(None, 2m)
    #([ht-1, st-1])Wd,Wd大小(2m,n)
    temp1 = Dense(n, use_bias=False)(concat)   #(None,n)
    temp2 = RepeatVector(N)(temp1)  #(None,N,n)
    #h_enUd,Ud大小(n,n)
    temp3 = Dense(n, use_bias=False)(h_en_all) #(None,N,n)
    #([ht-1, st-1])Wd+h_enUd
    temp4 = Add()([temp2, temp3])  #(None,N,n)
    #tanh(([ht-1, st-1])Wd+h_enUd)
    temp5 = Activation(activation='tanh')(temp4)  #(None,N,n)
    #(([ht-1, st-1])Wd+h_enUd)vd,vd大小(n*1)
    temp6 = Dense(1, use_bias=False)(temp5) #(None,N,1)
    beta = Activation(activation='softmax')(temp6) #(None,N,1)
    context = Dot(axes = 1)([beta, h_en_all])  #(None,1,n)
    
    return context

def Decoder_Attention(Y0, Y, s0_de, h0_de, h_en_all):
    M = K.int_shape(Y)[1]
    n_K_de = K.int_shape(Y0)[2]
    m = K.int_shape(s0_de)[1]
    n = K.int_shape(h_en_all)[2]
    
    y_ini = Lambda(lambda y0: Y0[:, :, 0])(Y0) 
    for i in range(1, n_K_de):
        y_temp = Lambda(lambda y0: Y0[:, :, i])(Y0)
        y_ini = Concatenate()([y_ini, y_temp]) 
    
    y = y_ini #(None, n_K_de)
    s = s0_de #(None, m)
    h = h0_de #(None, m)
    
    y_output = None
    for t in range(M):
        #[x_p; ct]
        context = Decoder_Attention_one_step(h, s, h_en_all) #(None,1,n)
        y = Reshape((1, n_K_de))(y) #(None,1,n_K_de)
        y = Concatenate()([y, context]) #(None,1,n+n_K_de)
        #[x_p; ct]WD+bD
        Y_ = Dense(n_K_de, use_bias=True) (y) #(None,1,n_K_de)
        h, _, s = LSTM(m, return_state=True, name='de_lstm_'+str(t+1))(Y_, initial_state=[h, s]) #(None,m)
        #[hdt;ct]
        context = Reshape((n,))(context) #(None,n)
        h_ = Concatenate()([h, context]) #(None,m+n)
        #sigmoid(Wv[hdt;ct]+bv)
        y_p = Dense(1, use_bias=True, activation='sigmoid', name = 'de_dense'+str(t+1))(h_) #(None,1)
        if t != 0:
            y_output = Concatenate(axis =1 )([y_output, y_p])
        else:
            y_output = y_p
        
        #未来降雨与预测流量拼接
        y = Lambda(lambda y: Y[:, t, :])(Y) #(None,1)
        y = Concatenate()([y, y_p]) #(None,n_K_de)
    y_output = Reshape((M,1))(y_output) #(None,M,1)

    return y_output

#DSA-LSTM
def DSA_LSTM(N, M, n_K_en, n_K_de, n, m):
    X = Input(shape=(N, n_K_en)) #输入时间序列数据,各站点雨量和流域出口流量
    s0_en = Input(shape=(n,)) #Encoder记忆单元状态初始化
    h0_en = Input(shape=(n,)) #Encoder隐藏单元状态初始化
    
    #Encoder部分-------------------------------------------------------------------
    X_ = Encoder_Attention(X, h0_en, s0_en)
    h_en_all = LSTM(n, return_sequences=True)(X_)

    #Decoder部分输出预测序列--------------------------------------------------------
    Y = Input(shape=(M, 1)) #未来M个时刻的降雨
    Y0 = Input(shape=(1, n_K_de)) #当前时刻的面雨量和出口流量
    s0_de = Input(shape=(m,)) #Decoder记忆单元状态初始化
    h0_de = Input(shape=(m,)) #Decoder隐藏单元状态初始化
    
    Q_output = Decoder_Attention(Y0, Y, s0_de, h0_de, h_en_all)
    model = Model(inputs=[X, Y, Y0, s0_en, h0_en, s0_de, h0_de], outputs=Q_output)
    
    return model