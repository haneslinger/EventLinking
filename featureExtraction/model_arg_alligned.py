import numpy as np
from keras.models import Model
from keras.layers.core import  Dropout, Reshape, Lambda
from keras.layers import Dense, Input, merge, Flatten, Concatenate, TimeDistributed
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv1D, Conv2D
from keras.models import model_from_json
from keras.models import load_model

import os
import json
#https://statcompute.wordpress.com/2017/01/08/an-example-of-merge-layer-in-keras/
#https://stackoverflow.com/questions/41603357/model-ensemble-with-shared-layers

def myfunc(x):

    print('at input {}'.format(x.shape))
    #for ten in x:
        #ten = x[i]
    temp1 = x[:][0:4]
    temp2 = x[:][5:]
    print('temp1= {} temp2={}'.format(temp1.shape, temp2.shape))
    x = merge([temp2, temp1], mode = 'concat')
    print('at output {}'.format(x.shape))
    return x

class My_Model:
    def __init__(self,input_shape, dimention):
        self.ip1 = Input(shape=(input_shape,1))
        self.ip2 = Input(shape=(input_shape,1))
        self.sim = Input(shape=(1,))
        print(self.ip1.shape)
        ##---projection layer-----##
        ## assume each argument is of size 4 and 5 argument per event. hence ip1 is 20x1. so is ip2self.
        ## after conv each argument will have 5 channles(dimention)
        self.projection1 = Conv1D(5, 4, strides=4, padding='valid', activation='tanh')#(Dropout(0.5)(Dense(1500, activation='relu')))
        self.sh_projection1_op1 = self.projection1(self.ip1)
        self.sh_projection1_op2 = self.projection1(self.ip2)

        print(self.sh_projection1_op1.shape)
        self.f_v1 = Flatten()(self.sh_projection1_op1)
        self.f_v2 = Flatten()(self.sh_projection1_op2)
        print('after flatten {}'.format(self.f_v1.get_shape().as_list()))
        self.projection2 = Conv1D(1, 5, strides=5, padding='valid', activation='tanh')
        self.allignment =list()

        x0= self.f_v2
        x1 = Lambda(myfunc,output_shape=self.f_v2.get_shape().as_list()[1])(x0)
        x2 = Lambda(myfunc,output_shape=self.f_v2.get_shape().as_list()[1])(x1)
        x3 = Lambda(myfunc,output_shape=self.f_v2.get_shape().as_list()[1])(x2)
        x4 = Lambda(myfunc,output_shape=self.f_v2.get_shape().as_list()[1])(x3)

        print('*****@@@@@+++++{} and {} and {} and {} and{}'.format(x0.shape, x1.shape, x2.shape, x3.shape, x4.shape))

        self.merged_layer = merge([self.f_v1, x0], mode = 'concat')
        self.merged_layer = Reshape((-1,2))(self.merged_layer)
        allgn = self.projection2(self.merged_layer)
        self.allignment.append(allgn)

        self.merged_layer = merge([self.f_v1, x1], mode = 'concat')
        self.merged_layer = Reshape((-1,2))(self.merged_layer)
        allgn = self.projection2(self.merged_layer)
        self.allignment.append(allgn)

        self.merged_layer = merge([self.f_v1, x2], mode = 'concat')
        self.merged_layer = Reshape((-1,2))(self.merged_layer)
        allgn = self.projection2(self.merged_layer)
        self.allignment.append(allgn)

        self.merged_layer = merge([self.f_v1, x3], mode = 'concat')
        self.merged_layer = Reshape((-1,2))(self.merged_layer)
        allgn = self.projection2(self.merged_layer)
        self.allignment.append(allgn)

        self.merged_layer = merge([self.f_v1, x4], mode = 'concat')
        self.merged_layer = Reshape((-1,2))(self.merged_layer)
        allgn = self.projection2(self.merged_layer)
        self.allignment.append(allgn)
        '''
        for i in range(5):
            self.merged_layer = merge([self.f_v1, self.f_v2], mode = 'concat')
            self.merged_layer = Reshape((-1,2))(self.merged_layer)
            allgn = self.projection2(self.merged_layer)
            #print('{} dim merged layer{} dim of conv allgn{}'.format(i,self.merged_layer.shape,allgn.shape ))
            self.allignment.append(allgn)
            temp1 = self.f_v2[0:4]
            temp2 = self.f_v2[5:]
            self.f_v2 = merge([temp2, temp1], mode = 'concat')
            #print(self.f_v2.type)
        '''
        #print(self.allignment[0].shape)
        for i in range(len(self.allignment)):
            self.allignment[i] = Reshape((5,1))(self.allignment[i])
            #print('{} dim allgnlayer{}'.format(i,self.allignment[i] ))

        self.allignment_all = merge(self.allignment, mode = 'concat')

        self.prediction = Dense(1, activation='sigmoid')(Flatten()(self.allignment_all))

        self.model = Model(input=[self.ip1, self.ip2, self.sim], output=self.prediction)
        sgd = SGD(lr = 0.1, momentum = 0.9, decay = 0, nesterov = False)
        self.model.compile(loss = 'mean_squared_error', optimizer = sgd, metrics = ['accuracy'])
        #seed(2017)
        #self.model.fit([X1, X2], Y.values, batch_size = 2000, nb_epoch = 100, verbose = 1)
    def train_model(self,train_X1,train_X2, train_S, train_y,batch_size_=50,epch=15):
        self.model.fit([train_X1,train_X2,train_S], train_y, batch_size=batch_size_, nb_epoch=epch, verbose=1, shuffle=True)
        self.epch = epch

    def predict(self,test_x1,test_x2, train_S):
        return self.model.predict([test_x1,test_x2, train_S])

    def evaluate(self,X1,X2,S,y):
        return self.model.evaluate([X1,X2,S],y)

    def save_Model_separately(self, path):
        weights_file = os.path.join(path,'weight.h5')
        model_file = os.path.join(path, 'model_archi.json')
        # Save the weights
        self.model.save_weights(weights_file)
        with open(model_file, 'w') as f:
            f.write(self.model.to_json())

    def save_model(self, path):
        model_file = os.path.join(path,'model_FM'+str(self.epch)+'.h5')
        self.model.save(model_file)
        return str(model_file)

    def load_model(self, path):
        self.model= load_model(path)



if __name__ == '__main__':
    train_x1 = np.random.rand(100,20)
    train_x2 = np.random.rand(100,20)
    print(train_x1.shape)
    print(train_x2.shape)
    train_x1 = np.expand_dims(train_x1, axis=2)
    train_x2 = np.expand_dims(train_x2, axis=2)
    print(train_x1.shape)
    print(train_x2.shape)
    train_y = np.random.randint(2, size=train_x1.shape[0])
    train_sim = np.random.rand(100)
    print(train_y.shape)
    model = My_Model(train_x1.shape[1],50)
    model.train_model(train_x1,train_x2,train_sim,train_y,epch=2)
