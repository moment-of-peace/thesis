'''
generate corresponding results for training documents
Author: Yi Liu
'''
import numpy as np
import pickle
import keras.layers as kl
import keras.models as km
import sys
import getopt

import my_utils as util
import gen_dataset as gen

def trainModel(trainData, trainResult, embedModel, epoch):
    # build and fit model
    model = km.Sequential()
    model.add(kl.Embedding(embedModel.shape[0],embedModel.shape[1], mask_zero=True,weights=[embedModel]))
    model.add(kl.Bidirectional(kl.LSTM(20,activation='relu',return_sequences=True)))
    model.add(kl.Bidirectional(kl.LSTM(20, return_sequences=True)))
    model.add(kl.TimeDistributed(kl.Dense(19)))
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
    weightsPath = 'weights_made_8000.npy'
    vocabPath = 'vocab_made_8000.pkl'
    trainPath = '__data__/MADE-1.0/process2_stepFour_corp'
    truthPath = '__data__/MADE-1.0/process2_stepThree_entity'
    windowSize = 20
    epoch = 30

    # parse arguments
    options,args = getopt.getopt(sys.argv[1:],"w:v:x:y:s:S:W:e:")
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
        
    # load weights and vocabulary
    embedModel = np.load(weightsPath)
    with open(vocabPath, 'rb') as handle:   # load vocabulary from file
        vocab = pickle.load(handle)
    # cross validation
    flist = util.sorted_file_list(trainPath, cmp_file)
    flist = gen.cycleShift(flist, shift, shiftSize)
    # generate x and y
    trainx = gen.getIndex(trainPath, flist, vocab)
    #trainy = gen.genResult(truthPath,flist)
    trainy = gen.genResultBin(truthPath, flist)
    # split data for training and testing respectively
    trainData,trainResult = gen.genTrainDataset(trainx[shiftSize:], trainy[shiftSize:], windowSize, 10)
    testData, testResult = gen.genTrainDataset(trainx[0:shiftSize], trainy[0:shiftSize], windowSize, windowSize)
    # train, predict, and evaluate
    model = trainModel(np.array(trainData), np.array(trainResult), embedModel, epoch)
    model.save('model_made_%d-%d_%d-epoch.h5'%(shiftSize, shift, epoch))
    predict = model.predict(np.array(testData))
    pred, tru = evalModel(predict, np.array(testResult), shiftSize, shift, epoch)

    #restore
    #gen.restore(flist[0:shiftSize], pred)

if __name__ == '__main__':
    main()