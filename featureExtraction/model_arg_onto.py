import numpy as np
from keras.models import Model
from keras.layers.core import  Dropout, Reshape, Lambda
from keras.layers import Dense, Input, merge, Flatten, concatenate, TimeDistributed, dot
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv1D, Conv2D
from keras.models import model_from_json
from keras.models import load_model

import os
import json
#https://statcompute.wordpress.com/2017/01/08/an-example-of-merge-layer-in-keras/
#https://stackoverflow.com/questions/41603357/model-ensemble-with-shared-layers

class My_Model:
    def __init__(self,ev_input_shape, arg_input_shape, onto_shape, dimention):
        self.arg_ip1 = Input(shape=(arg_input_shape,1))
        self.arg_ip2 = Input(shape=(arg_input_shape,1))
        self.ev_ip1 = Input(shape=(ev_input_shape,))
        self.ev_ip2 = Input(shape=(ev_input_shape,))
        self.onto_ip = Input(shape=(onto_shape,))
        #self.sim = Input(shape=(1,))
        #print(self.ip1.shape)
        ##---projection layer-----##
        ## assume each argument is of size 5 and 5 arguments per event. hence ip1 is 20x1. so is ip2self.
        ## after conv each argument will have 5 channles(dimention)
        self.projection1 = Conv1D(5, 4, strides=5, padding='valid', activation='elu')#(Dropout(0.5)(Dense(1500, activation='relu')))
        self.sh_projection1_op1 = self.projection1(self.arg_ip1)
        self.sh_projection1_op2 = self.projection1(self.arg_ip2)

        #print(self.sh_projection1_op1.shape)
        #self.merged_layer = concatenate([self.sh_projection1_op1, self.sh_projection1_op2], axis=1)
        #print(self.merged_layer.shape)
        self.projection2 = TimeDistributed(Dense(8,activation='elu'), input_shape=(self.sh_projection1_op1.shape[1], self.sh_projection1_op1.shape[2]))#(self.merged_layer
        self.sh_projection2_op1 = self.projection2(self.sh_projection1_op1)
        self.sh_projection2_op2 = self.projection2(self.sh_projection1_op2)
        #print(self.sh_projection2_op2.shape)

        self.inter_allgn = dot([self.sh_projection2_op1, self.sh_projection2_op2],axes=-1)
        self.intra_allgn1 = dot([self.sh_projection2_op1, self.sh_projection2_op1],axes=-1)
        self.intra_allgn2 = dot([self.sh_projection2_op2, self.sh_projection2_op2],axes=-1)
        #print(self.intra_allgn1.shape)

        self.inter_allgn_f = Flatten()(self.inter_allgn)
        self.intra_allgn1_f = Flatten()(self.intra_allgn1)
        self.intra_allgn2_f = Flatten()(self.intra_allgn2)
        #print(self.intra_allgn1_f.shape)

        self.all_allgn = concatenate([self.inter_allgn_f,self.intra_allgn1_f,self.intra_allgn2_f])
        #print(self.all_allgn.shape)
        #--- event vector processing----#
        print(self.ev_ip1.shape)
        #self.ev_vec = Reshape((2,int(self.ev_ip1.shape[1])))(concatenate([self.ev_ip1, self.ev_ip2]))
        #self.ev_projection = TimeDistributed(Dense(8,activation='relu'), input_shape=(self.ev_vec.shape[1], self.ev_vec.shape[2]))(self.ev_vec)
        self.ev_projection = Dense(8,activation='elu')
        self.ev1 = self.ev_projection(self.ev_ip1)
        self.ev2 = self.ev_projection(self.ev_ip2)
        print(self.ev1.shape)
        self.ev_score = dot([self.ev1,self.ev2], axes=-1)

        self.onto_vec = Dense(1,activation = 'sigmoid')(Dense(5, activation='elu')(self.onto_ip ))
        self.all_allgn = concatenate([self.all_allgn, self.ev_score, self.onto_vec])
        print(self.ev_score.shape)
        #self.ev_projection1 = Dense(5, activation)

        self.vector = Dense(dimention)(self.all_allgn)
        self.prediction = Dense(1, activation='sigmoid')(self.all_allgn)

        self.model = Model(input=[self.ev_ip1, self.arg_ip1, self.ev_ip2, self.arg_ip2,self.onto_ip], output=self.prediction)
        sgd = SGD(lr = 0.1, momentum = 0.9, decay = 0, nesterov = False)
        self.model.compile(loss = 'mean_squared_error', optimizer = sgd, metrics = ['accuracy'])
        #seed(2017)
        #self.model.fit([X1, X2], Y.values, batch_size = 2000, nb_epoch = 100, verbose = 1)
    def train_model(self,train_X1, arg_X1, train_X2, arg_X2, onto_ip, train_y,batch_size_=50,epch=15):
        self.model.fit([train_X1, arg_X1, train_X2, arg_X2,onto_ip], train_y, batch_size=batch_size_, nb_epoch=epch, verbose=1, shuffle=True)
        self.epch = epch

    def predict(self,test_x1,arg_x1,test_x2, arg_x2, onto_ip ):
        return self.model.predict([test_x1, arg_x1, test_x2, arg_x2,onto_ip])

    def evaluate(self,X1,A1, X2, A2, ontoip, y):
        return self.model.evaluate([X1,A1,X2,A2,ontoip,y)

    def save_Model_separately(self, path):
        weights_file = os.path.join(path,'weight.h5')
        model_file = os.path.join(path, 'model_archi.json')
        # Save the weights
        self.model.save_weights(weights_file)
        with open(model_file, 'w') as f:
            f.write(self.model.to_json())

    def save_model(self, path):
        model_file = os.path.join(path,'model_arg_onto_'+str(self.epch)+'.h5')
        self.model.save(model_file)
        return str(model_file)

    def load_model(self, path):
        self.model= load_model(path)



if __name__ == '__main__':
    train_x1 = np.random.rand(100,20)
    train_x2 = np.random.rand(100,20)
    arg_x1 = np.random.rand(100,20)
    arg_x2 = np.random.rand(100,20)
    print(train_x1.shape)
    print(train_x2.shape)
    #train_x1 = np.expand_dims(train_x1, axis=2)
    #train_x2 = np.expand_dims(train_x2, axis=2)
    arg_x1 = np.expand_dims(arg_x1, axis=2)
    arg_x2 = np.expand_dims(arg_x2, axis=2)
    print(train_x1.shape)
    print(train_x2.shape)
    train_y = np.random.randint(2, size=train_x1.shape[0])
    train_sim = np.random.rand(100)
    print(train_y.shape)
    model = My_Model(train_x1.shape[1], arg_x1.shape[1],50)
    model.train_model(train_x1, arg_x1, train_x2, arg_x2, train_y,epch=2)
    py= model.predict(train_x1, arg_x1, train_x2, arg_x2)
