'''
generate corresponding results for training documents
Author: Yi Liu
'''
import numpy as np
import pickle
import keras.layers as kl
import keras.models as km
import keras.optimizers as opt
from sklearn.model_selection import KFold
#from keras_contrib.layers.crf import CRF

import sys
import getopt

import my_utils as util
import gen_dataset as gen

def trainModel(trainData, trainResult, embedModel, epoch, outputsize):
    # build and fit model
    model = km.Sequential()
    model.add(kl.Embedding(embedModel.shape[0],embedModel.shape[1], mask_zero=True,weights=[embedModel]))
    model.add(kl.Bidirectional(kl.LSTM(20,activation='relu',return_sequences=True))) # GRU?
    model.add(kl.Bidirectional(kl.LSTM(20,activation='relu',return_sequences=True)))
    model.add(kl.TimeDistributed(kl.Dense(outputsize)))
    #crf_layer = CRF(outputsize)
    #model.add(crf_layer)
    #model.compile('rmsprop', loss=crf_layer.loss_function, metrics=[crf_layer.accuracy])
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainData, trainResult, epochs=epoch, batch_size=100, verbose=2)
    return model

# evaluate
def evalModel(predict, testResult, shiftSize, shift, epoch):
    n = 0
    tru = []
    pred = []
    shape = predict.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            p = util.maxIndex(predict[i][j])
            t = util.maxIndex(testResult[i][j])
            pred.append(p)
            tru.append(t)
            if p == t:
                n += 1
    print('ave accurate: %f'%(n/shape[0]/shape[1]))
    with open('result_%d_%d_%d.txt'%(shiftSize, shift,epoch),'wt') as tar:
        for i in range(len(tru)):
            tar.write('%d %d\n'%(tru[i],pred[i]))
    return tru, pred

def cmp_file(x):
    c = x.split('_')
    return int(c[0]) * 100000 + int(c[1])

def main():
    shift = 0
    shiftSize = 90
    weightsPath = 'weights_nodiscard_8000.npy'
    vocabPath = 'vocab_nodiscard_8000.pkl'
    #weightsPath = 'weights_glove_400000.npy'
    #vocabPath = 'vocab_glove_400000.pkl'
    trainPath = '__data__/MADE2-1.0/process2_stepFour_corp'
    truthPath = '__data__/MADE2-1.0/process2_stepThree_entity'
    windowSize = 20
    epoch = 30
    nclass = -1
    outputsize = 19
    maxFold = 10
    folds = 10
    

    # parse arguments
    options,args = getopt.getopt(sys.argv[1:],"w:v:x:y:s:S:W:e:c:m:f:")
    for opt, para in options:
        if opt == '-w':
            weightsPath = para
        if opt == '-v':
            vocabPath = para
        if opt == '-x':
            trainPath = para
        if opt == '-y':
            truthPath = para
        if opt == '-s':
            shift = int(para)
        if opt == '-S':
            shiftSize = int(para)
        if opt == '-W':
            windowSize = int(para)
        if opt == '-e':
            epoch = int(para)
        if opt == '-c':
            nclass = int(para)
            outputsize = 3
        if opt == '-m':
            maxFold = int(para)
        if opt == '-f':
            folds = int(para)
        
    # load weights and vocabulary
    embedModel = np.load(weightsPath)
    with open(vocabPath, 'rb') as handle:   # load vocabulary from file
        vocab = pickle.load(handle)
    
    
    flist = util.sorted_file_list(trainPath, cmp_file)
    print(flist[:10])
    #flist = gen.cycleShift(flist, shift, shiftSize)
    # generate x and y
    x = gen.getIndex(trainPath, flist, vocab)
    #y = gen.genResult(truthPath,flist)
    y = gen.genResultBin(truthPath, flist)
    # convert to two-class
    if nclass != -1:
        y = gen.twoClass(y, [nclass,nclass+1])
    # k-folds using scikit learn
    datax, datay = gen.toSent(flist, trainPath, x, y)
    x_pad, y_pad = gen.padSent(datax, datay, 100)
    if folds > 1:
        kf = KFold(n_splits=folds, shuffle=True)
        fold = 0
        
        for train_index, test_index in kf.split(x_pad):
            trainData, trainResult = [x_pad[i] for i in train_index], [y_pad[i] for i in train_index]
            testData, testResult = [x_pad[i] for i in test_index], [y_pad[i] for i in test_index]
            print(len(trainData),len(trainResult),len(testData),len(testResult))
            
            # train, predict, and evaluate
            model = trainModel(np.array(trainData), np.array(trainResult), embedModel, epoch, outputsize)
            model.save('model_made_%d-%d_%d-epoch.h5'%(shiftSize, fold, epoch))
            predict = model.predict(np.array(testData))
            pred, tru = evalModel(predict, np.array(testResult), shiftSize, fold, epoch)
                
            fold += 1
            if fold == maxFold:
                break
    else:
        #np.save('trainx_%d-epoch'%(epoch),np.array(x_pad))
        #np.save('trainy_%d-epoch'%(epoch),np.array(y_pad))
        model = trainModel(np.array(x_pad), np.array(y_pad), embedModel, epoch, outputsize)
        model.save('model_made_%d-epoch.h5'%(epoch))
    '''    
    
    # cross validation
    flist = util.sorted_file_list(trainPath, cmp_file)
    util.shiftFiles(flist, trainPath, '%s%d'%(trainPath, shift), shiftSize, shift)
    util.shiftFiles(flist, truthPath, '%s%d'%(truthPath, shift), shiftSize, shift, tail='.npy')
    trainPath, truthPath = '%s%d'%(trainPath, shift), '%s%d'%(truthPath, shift)
    
    flist = util.sorted_file_list(trainPath, cmp_file)
    print(flist[:10])
    #flist = gen.cycleShift(flist, shift, shiftSize)
    # generate x and y
    x = gen.getIndex(trainPath, flist, vocab)
    #y = gen.genResult(truthPath,flist)
    y = gen.genResultBin(truthPath, flist)
    
    # split data by sentences
    trainx, trainy = gen.toSent(flist[shiftSize:], trainPath, x[shiftSize:], y[shiftSize:])
    testx, testy =  gen.toSent(flist[:shiftSize], trainPath, x[:shiftSize], y[:shiftSize])
    # pad to same length
    trainData,trainResult = gen.padSent(trainx, trainy, 100)
    testData, testResult = gen.padSent(testx, testy, 100)
    print(len(trainData),len(trainResult),len(testData),len(testResult))
    
    # split data for training and testing respectively
    #trainData,trainResult = gen.genTrainDataset(x[shiftSize:], y[shiftSize:], windowSize, 10)
    #testData, testResult = gen.genTrainDataset(x[0:shiftSize], y[0:shiftSize], windowSize, windowSize)
    
    
    # train, predict, and evaluate
    model = trainModel(np.array(trainData), np.array(trainResult), embedModel, epoch, outputsize)
    model.save('model_made_%d-%d_%d-epoch.h5'%(shiftSize, shift, epoch))
    predict = model.predict(np.array(testData))
    pred, tru = evalModel(predict, np.array(testResult), shiftSize, shift, epoch)

    #restore
    #gen.restore(flist[0:shiftSize], pred)
    '''

if __name__ == '__main__':
    main()
 