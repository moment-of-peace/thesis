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
import os

import my_utils as util

# convert words in files into index, return a list of different length lists
def getIndex(path, filelist, vocab):
    data = []
    for f in filelist:
        data.append(indexFile(os.path.join(path, f), vocab))
    return data
# convert words in a single file into index, return a list of integers
def indexFile(src, vocab):
    index = []
    with open(src, 'rt') as f:
        content = f.read().split(' ')
    for word in content:
        try:
            index.append(vocab[word])
        except KeyError:
            index.append(vocab['UNK'])
    return index

# return a list ("results") of lists, and each nested list ("entities") is a list of entity vectors
def genResult(path, flist):
    # map entities to vectors
    entityDict = {'a':[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  'i':[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  's':[0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                  'v':[0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                  'd':[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  'o':[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                  'r':[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                  'f':[0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                  'u':[0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  'x':[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}
    results = []
    for f in flist:
        entities = []
        with open(os.path.join(path, f), 'rt') as src:
            content = src.read().split(' ')
        for word in content:
            entities.append(entityDict[word[0].lower()])
        results.append(entities)
    return results

# use window method to cut datasets for training
def genTrainDataset(trainx, trainy, windowsize, step):
    data = []
    result = []
    for i in range(0, len(trainx)):
        data.extend(window(trainx[i],windowsize,step))
        result.extend(window(trainy[i],windowsize,step))
    return data, result 
def window(data, windowsize, step):
    result = []
    for i in range(0, len(data)-windowsize, step):
        result.append(data[i:i+windowsize])
    result.append(data[-windowsize:])
    return result

# do a cycling shift on a list
def cycleShift(flist, shift, shiftSize):
    assert(shift*shiftSize < len(flist))
    shiftedList = flist[shift*shiftSize:]
    shiftedList.extend(flist[0:shift*shiftSize])
    return shiftedList

def trainModel(trainData, trainResult, embedModel, epoch):
    # build and fit model
    model = km.Sequential()
    model.add(kl.Embedding(embedModel.shape[0],embedModel.shape[1], mask_zero=True,weights=[embedModel]))
    model.add(kl.Bidirectional(kl.LSTM(20,activation='relu',return_sequences=True)))
    model.add(kl.Bidirectional(kl.LSTM(20, return_sequences=True)))
    model.add(kl.TimeDistributed(kl.Dense(10)))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainData, trainResult, epochs=epoch, batch_size=100, verbose=2)
    return model

# the index of max element in a array
def maxIndex(l):
    max = l[0]
    index = [0]
    for i in range(1,l.shape[0]):
        if l[i] > max:
            index = [i]
            max = l[i]
        elif l[i] == max:
            index.append(i)
    sum = 0
    for e in index:
        sum += e
    return round(sum/len(index))

# predict and evaluate
def evalModel(testData, testResult, model, shiftSize, shift, epoch):
    predict = model.predict(testData)
    n = 0
    tru = []
    pred = []
    shape = predict.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            p = maxIndex(predict[i][j])
            t = maxIndex(testResult[i][j])
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
    trainPath = '__data__/MADE-1.0/process_stepFour_corp'
    truthPath = '__data__/MADE-1.0/process_stepThree_entity'
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

    flist = util.sorted_file_list(trainPath, cmp_file)
    flist = cycleShift(flist, shift, shiftSize)
    trainx = getIndex(trainPath, flist, vocab)
    trainy = genResult(truthPath,flist)
    trainData,trainResult = genTrainDataset(trainx[shiftSize:], trainy[shiftSize:], windowSize, 10)
    #trainData = drop(trainData, int(windowSize/4))
    #trainResult = drop(trainResult, int(windowSize/4))
    testData, testResult = genTrainDataset(trainx[0:shiftSize], trainy[0:shiftSize], windowSize, windowSize)

    #trainData, trainResult = np.load('tx.npy'), np.load('ty.npy')
    #testData, testResult = np.load('testx.npy'), np.load('testy.npy')
    model = trainModel(trainData, trainResult, embedModel, epoch)
    evalModel(testData, np.array(testResult), model, shiftSize, shift, epoch)

if __name__ == '__main__':
    main()