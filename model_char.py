'''
generate corresponding results for training documents
Author: Yi Liu
'''
import numpy as np
import os
import pickle
import yaml

import keras.layers as kl
import keras.models as km
import keras.optimizers as opt
import keras.callbacks as kc
from sklearn.model_selection import KFold
#from keras_contrib.layers.crf import CRF

import sys
import getopt

import my_utils as util
import gen_dataset as gen
from toxml import mainProcess

class NERModel(object):
    def __init__(self, p):
        # load weights and vocabulary
        self.wei_token = np.load(p['weightsPath'])
        self.wei_char = util.gen_char_weights(p['char_dim'])
        
        self.vocab_size = self.wei_token.shape[0]
        self.token_dim = self.wei_token.shape[1]
        self.char_size = self.wei_char.shape[0]
        self.char_dim = self.wei_char.shape[1]
        self.filters = int(self.token_dim/4)
        self.model = None
    
    def build_model(self, p):
        # token level
        inp_token = kl.Input(shape=(p['sent_len'],))
        embed_token = kl.Embedding(self.vocab_size,self.token_dim,mask_zero=True,weights=[self.wei_token])
        embed_token.trainable = False
        embed_tout = embed_token(inp_token)
        inp_char = None
        if p['use_char']:
            # char level
            inp_char = kl.Input(shape=(p['sent_len'], p['word_len']))
            embed_char = kl.Embedding(self.char_size,self.char_dim,weights=[self.wei_char])
            embed_char.trainable = False
            embed_cout = embed_char(inp_char)
            # convolutional layer
            print('conv filters:', self.filters)
            conv_layer = kl.Conv2D(self.filters,(1,3),input_shape=(p['sent_len'],p['word_len'],self.char_dim),use_bias=False,padding='SAME')
            conv_out = conv_layer(embed_cout)
            # maxpooling
            pool_layer = kl.MaxPooling2D(pool_size=(1,p['word_len']))
            pool_out = pool_layer(conv_out)
            reshape_out = kl.Reshape((p['sent_len'], self.filters))(pool_out)
            
            # concatenation
            concat_out = kl.concatenate([embed_tout,reshape_out],axis=2)
        else:
            concat_out = embed_tout
        lstm_out = kl.Bidirectional(kl.LSTM(p['units'],activation=p['lstm_act'],return_sequences=True,dropout=0.4))(concat_out)
        dense_out = kl.TimeDistributed(kl.Dense(p['outputsize']))(lstm_out)
        
        # build model
        if p['use_char']:
            model_ner = km.Model(inputs=[inp_token,inp_char],outputs=dense_out)
        else:
            model_ner = km.Model(inputs=inp_token,outputs=dense_out)
        model_ner.compile(loss=p['lstm_loss'], optimizer=p['lstm_opt'])
        model_ner.summary()
        self.model = model_ner
       
    # set char_indices to None if not using char level embedding
    def train_model(self, p, token_indices, char_indices, train_result):
        # call back functions
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        earlyStop = kc.EarlyStopping(monitor='loss', mode='min', patience=3)
        checkpoint = kc.ModelCheckpoint('checkpoints/checkpoint.{epoch:02d}-{loss:.2f}.h5', monitor='loss', mode='min', period=2)
        callback = [checkpoint,earlyStop]
        # fit model
        if p['use_char']:
            train_data = [token_indices,char_indices]
        else:
            train_data = token_indices
        self.model.fit(train_data, train_result, epochs=p['epoch'], batch_size=p['batch_size'], verbose=2, callbacks=callback)

def cmp_file(x):
    c = x.split('_')
    return int(c[0]) * 100000 + int(c[1])

def main():
    yml_path = 'params.yml'
    with open(yml_path, 'r', encoding='utf-8') as f:
        cont = f.read()
    params = yaml.load(cont)
    
    with open(params['vocabPath'], 'rb') as handle:   # load vocabulary from file
        vocab = pickle.load(handle)
    
    flist = util.sorted_file_list(params['trainPath'], cmp_file)
    print(flist[:10])
    # generate x and y
    x = gen.getIndex(params['trainPath'], flist, vocab)
    #y = gen.genResult(truthPath,flist)
    y = gen.genResultBin(params['truthPath'], flist)
    # convert to two-class
    if params['nclass'] != -1:
        y = gen.twoClass(y, [params['nclass'],params['nclass']+1])
    # k-folds using scikit learn
    datax, datay = gen.toSent(flist, params['trainPath'], x, y)
    x_pad, y_pad = gen.padSent(datax, datay, params['sent_len'])
    
    if params['use_char']:
        char_indices = np.array(gen.gen_char_indices(x_pad,vocab))
    else:
        char_indices = None
        
    model_ner = NERModel(params)
    model_ner.build_model(params)
    
    if params['folds'] > 1:
        kf = KFold(n_splits=params['folds'], shuffle=True)
        fold = 0
        
        for train_index, test_index in kf.split(x_pad):
            train_data, train_result = [x_pad[i] for i in train_index], [y_pad[i] for i in train_index]
            test_data, test_result = [x_pad[i] for i in test_index], [y_pad[i] for i in test_index]
            print(len(train_data),len(train_result),len(test_data),len(test_result))
            '''
            # train, predict, and evaluate
            model = trainModel(np.array(train_data), np.array(train_result), wei_token, epoch, outputsize, units, batch_size)
            model.save('model_made_%d-%d_%d-epoch.h5'%(shiftSize, fold, epoch))
            predict = model.predict(np.array(test_data))
            pred, tru = evalModel(predict, np.array(test_result), shiftSize, fold, epoch)
            '''
            fold += 1
            if fold == maxFold:
                break
    else:
        print('input shape',char_indices.shape,np.array(x_pad).shape)
        model_ner.train_model(params, np.array(x_pad), char_indices, np.array(y_pad))
        mainProcess(model_ner.model, vocabPath, load_file=False, cut=params['sent_len'])
        try:
            model_file = 'model_made_%d-epoch.h5'%(epoch)
            model_ner.model.save(model_file)
        except:
            print('cannot save')

if __name__ == '__main__':
    main()
