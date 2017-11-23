'''
generate corresponding results for training documents
Author: Yi Liu
'''
import numpy as np
import pickle
import keras.layers as kl
import keras.models as km
#import gensim
#import os
import sys
import getopt

import preprocessing as pre

global model

def loadW2VModel(modelFile):
    global model
    model = gensim.models.KeyedVectors.load_word2vec_format(modelFile, binary=False)

def gen_embed_model(modelFile):
    vocab = {}  # {'word': index, ...}
    with open(modelFile,'r') as f:
        line = f.readline()
        [length, dim] = line.split(' ')
        vec = np.zeros((int(length)+1, int(dim)), dtype = np.float64)    # {index: [vector], ...}
        line = f.readline()
        i = 1
        while line != '':
            index = line.find(' ')
            word = line[:index]
            vector = []
            for e in line[index+1:].split(' '):
                try:
                    vector.append(float(e))
                except Exception:
                    print('float' + e)
            vocab[word] = i
            vec[i] = np.array(vector)
            line = f.readline()
            i = i+1
    return [vocab, vec]

def extract_true_entity(srcFile):
    '''
    a nested dictionary used to store all entities and their position
    {lineNum 1: {position 1: entity 1, position 2: entity 2, ...}, lineNum 2: ... ...}
    -1: [start, entity], -2: [end, entity], -3: entity
    '''
    entities = dict()
    with open(srcFile, 'r') as f:
        for line in f:
            # each entity is seperated by '||'
            iterms = line.strip('\n').split('||')
            for e in iterms:
                add_entity(e, entities,srcFile)
    return entities

# add a single entity to the dictionary
def add_entity(item, entities,srcFile):
    index = item.find('=')
    name = item[:index]
    content = item[index+1:]
    # ignore "nm" and 'ln' entity
    if content != '\"nm\"' and name != 'ln':
        content = content.split('\" ')[1]
        content = content.strip(' ')
        for position in content.split(','):
            try:
                parse_position(position, entities, name)
            except Exception:
                print(srcFile, position)
        '''
        if ',' in content:
            [part1, part2] = content.split(',')
            start = part1.split(' ')[0]
            end = part2.split(' ')[1]
            [line1,position1] = start.split(':')
            [line2,position2] = end.split(':')
            line1 = int(line1)
            line2 = int(line2)
            if line1 in entities.keys():
                entities[line1][-2] = [int(position1), name]
            else:
                entities[line1] = {-2: [int(position1), name]}

            if line2 in entities.keys():
                entities[line2][-1] = [int(position2), name]
            else:
                entities[line2] = {-1: [int(position2), name]}
        else:
            temp = content.split(' ')
            start = int(temp[0].split(':')[1])  # start position
            end = int(temp[1].split(':')[1])    # end position
            line = int(temp[0].split(':')[0])   # line number
            # add to dictionary
            if line in entities.keys():
                entities[line][start] = 'b_' + name
                for i in range(start + 1, end + 1):
                    entities[line][i] = 'i_' + name
            else:
                entities[line] = {start: 'b_' + name}
                for i in range(start + 1, end + 1):
                    entities[line][i] = 'i_' + name
        '''

# parse an entity position seperated by ','
def parse_position(position, entities, name):
    temp = position.split(' ')
    [line, start] = temp[0].split(':')
    line, start = int(line), int(start) # start position
    [line2, end] = temp[1].split(':')
    line2, end = int(line2), int(end)   # end position
    # add to dictionary
    if line == line2:
        if line in entities.keys():
            entities[line][start] = 'b_' + name
            for i in range(start + 1, end + 1):
                entities[line][i] = 'i_' + name
        else:
            entities[line] = {start: 'b_' + name}
            for i in range(start + 1, end + 1):
                entities[line][i] = 'i_' + name
    else:   # a single entity is split into multiple lines
        if line in entities.keys():
            entities[line][-1] = [int(start), name]
        else:
            entities[line] = {-1: [int(start), name]}

        if line2 in entities.keys():
            entities[line2][-2] = [int(end), name]
        else:
            entities[line2] = {-2: [int(end), name]}

        for i in range(line+1, line2):
            if i in entities.keys():
                entities[i][-3] = name
            else:
                entities[i] = {-3: name}

# convert all tokens in a raw file into corresponding entities and write to a file
def match_file(rawFile, entities):
    newFile = rawFile + '.match'
    with open(rawFile, 'r') as src:
        with open(newFile, 'w') as out:
            i = 1
            for line in src:
                content = line.strip('\n').split(' ')
                for j in range(0, len(content)):
                    entity = 'ot'
                    # try whether there is a corresponding entity in the dictionary,
                    # if no, use 'ot', which means 'other'
                    try:
                        entity = entities[i][j]
                    except Exception:   # a single entity might be split into two lines
                        if i in entities.keys():
                            if -1 in entities[i] and j == entities[i][-1][0]:
                                entity = 'b_' + entities[i][-1][1]
                            elif -1 in entities[i] and j > entities[i][-1][0]:
                                entity = 'i_' + entities[i][-1][1]
                            elif -2 in entities[i] and j <= entities[i][-2][0]:
                                entity = 'i_' + entities[i][-2][1]
                            elif -3 in entities[i]:
                                entity = 'i_' + entities[i][-3]
                    out.write(entity + ' ')
                i = i + 1
                out.write('\r\n')
'''
# embed tokens in a file into vectors
def embed_file(srcFile, flag):
    global model
    result = []

    with open(srcFile, 'r') as src:
        for line in src:
            vector = []
            tokens = line.strip('\n').split(' ') # need handle non-letters
            for e in tokens:
                e = e.lower()
                try:
                    vector.append(model[e])
                except Exception:   # no corresponding word in w2v model
                    vector.append(model['a'])
            if flag == 'a':
                result.append(vector)
            else:
                result.extend(vector)
    return result
'''
# convert all tokens in a raw file into corresponding entities and return a list
def gen_train_result(path1, path2, filelist):
    # map entities to vectors
    entityDict = {'b_m':[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  'i_m':[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  'b_do':[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  'i_do':[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  'b_mo':[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  'i_mo':[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  'b_f':[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  'i_f':[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  'b_du':[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  'i_du':[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  'b_r':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  'i_r':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  'b_e':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                  'i_e':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                  'b_t':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  'i_t':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                  'b_c':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                  'i_c':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                  'b_ln':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  'ot':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}
    trainResult = []

    for f in filelist:
        entities = extract_true_entity(path2 + f)
        with open(path1+f, 'r') as src:
            result = []
            i = 0
            for line in src:
                i = i+1
                content = line.strip('\n').split(' ')
                for j in range(0, len(content)):
                    entity = 'ot'
                    # try whether there is a corresponding entity in the dictionary,
                    # if no, use 'ot', which means 'other'
                    try:
                        entity = entities[i][j]
                    except Exception:   # a single entity might be split into two lines
                        if i in entities.keys():
                            if -1 in entities[i] and j == entities[i][-1][0]:
                                entity = 'b_' + entities[i][-1][1]
                            elif -1 in entities[i] and j > entities[i][-1][0]:
                                entity = 'i_' + entities[i][-1][1]
                            elif -2 in entities[i] and j <= entities[i][-2][0]:
                                entity = 'i_' + entities[i][-2][1]
                            elif -3 in entities[i]:
                                entity = 'i_' + entities[i][-3]
                    #result.append(entityDict[entity])
                    result.append(entityDict[entity].index(1))
        trainResult.append(result)
    return trainResult

# generate a taining dataset from a directory
def gen_train_data(path, filelist):
    trainSet = []

    for f in filelist:
        trainSet.append(embed_file(path+f, 'e'))
    return trainSet

# generate index of train data
def get_index(path, filelist, vocab):
    trainData = []
    for f in filelist:
        trainData.append(index_file(path+f, vocab))
    return trainData
def index_file(srcFile, vocab):
    vector = []
    with open(srcFile,'r') as src:
        for line in src:
            tokens = line.strip('\n').split(' ') # need handle non-letters
            for e in tokens:
                e = e.lower()
                try:
                    vector.append(vocab[e])
                except:   # no corresponding word in w2v model
                    try:
                        vector.append(vocab[e])
                    except:
                        vector.append(vocab['a'])
    return vector

# find common files in two directory
def commonFiles(path1, path2):
    #path1 = input('input path1: \n')
    #path2 = input('input path2: \n')
    flist = []
    files = os.listdir(path1)
    for f in files:
        if os.path.exists(path2 + f):
            flist.append(f)
    return flist

def padding(x, y, length):
    #dim = x[0][0].shape
    for i in range(0, len(x)):
        for j in range(len(x[i]), length):
            #x[i].append(np.zeros(dim))
            #y[i].append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
            x[i].append(0)
            y[i].append(19)
    return [x, y]

# build a raw dataset containing a series of event type distribution
def build_dataset(data, start=0, end=None, dim=60):
    dataset = np.zeros((end-start, dim), dtype=np.int32)
    if end == None:
        end = len(flist)
    for i in range(start, end):
        dataset[i-start,:] = data[i]
    return dataset

# build dataset used for rnn training
def build_train_set(data, size, step, dim):
    trainSet = np.zeros((size, step, dim), dtype=np.float64)
    for i in range(size):
        trainSet[i,:,:] = build_dataset(data, i, i+step, dim)
    return trainSet

# build dataset used for rnn test
def build_train_result(data, size, step, dim):
    trainSet = np.zeros((size, dim), dtype=np.float64)
    for i in range(size):
        trainSet[i,:] = build_dataset(data, i+step, i+step+1, dim)
    return trainSet

def gen_train_dataset(trainx, trainy, windowsize):
    data = []
    result = []
    for i in range(len(trainx)):
        data.extend(window(trainx[i],windowsize))
        result.extend(window(trainy[i],windowsize))
    return data, result 
def window(data, windowsize):
    result = []
    for i in range(len(data)-windowsize):
        result.append(data[i:i+windowsize])
    return result

def drop(data, rate):
    result = []
    for i in range(len(data)):
        if i%rate == 0:
            result.append(data[i])
    return result

def train_model(trainData, trainResult, embedModel, epoch):
    # build and fit model
    model = km.Sequential()
    model.add(kl.Embedding(embedModel.shape[0],embedModel.shape[1], mask_zero=True,weights=[embedModel]))
    model.add(kl.Bidirectional(kl.LSTM(20,activation='relu',return_sequences=True)))
    model.add(kl.Bidirectional(kl.LSTM(20, return_sequences=True)))
    model.add(kl.TimeDistributed(kl.Dense(20)))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainData, trainResult, epochs=epoch, batch_size=100, verbose=2)
    return model

# do a cycling shift on a list
def cycleShift(flist, shift, shiftSize):
    assert(shift*shiftSize < len(flist))
    shiftedList = flist[shift*shiftSize:]
    shiftedList.extend(flist[0:shift*shiftSize])
    return shiftedList

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
def eval_model(testData, testResult, model, shiftSize, shift):
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
    with open('result_%d_%d.txt'%(shiftSize, shift),'wt') as tar:
        for i in range(len(tru)):
            tar.write('%d %d\n'%(tru[i],pred[i]))
    return tru, pred

def main():
    shift = 0
    shiftSize = 42
    weightsModelPath = 'weights_glove100.npy'
    vocabPath = 'vocab_glove100.pkl'
    trainPath = '2009train/'
    truthPath = '2009truth/'
    windowSize = 20
    epoch = 30

    # parse arguments
    options,args = getopt.getopt(sys.argv[1:],"w:v:x:y:s:S:W:e:")
    for opt, para in options:
        if opt == '-w':
            weightsModelPath = para
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
    embedModel = np.load(weightsModelPath)
    with open(vocabPath, 'rb') as handle:   # load vocabulary from file
        vocab = pickle.load(handle)

    flist = pre.sortedCommonFiles(trainPath,truthPath)
    flist = cycleShift(flist, shift, shiftSize)
    trainx = pre.get_index(trainPath, flist, vocab)
    trainy = pre.gen_train_result(trainPath,truthPath,flist)
    trainData,trainResult = pre.gen_train_dataset(trainx[shiftSize:], trainy[shiftSize:], windowSize, 1)
    trainData = pre.drop(trainData, int(windowSize/4))
    trainResult = pre.drop(trainResult, int(windowSize/4))
    testData, testResult = pre.gen_train_dataset(trainx[0:shiftSize], trainy[0:shiftSize], windowSize, windowSize)

    #trainData, trainResult = np.load('tx.npy'), np.load('ty.npy')
    #testData, testResult = np.load('testx.npy'), np.load('testy.npy')
    model = train_model(trainData, trainResult, embedModel, epoch)
    eval_model(testData, np.array(testResult), model, shiftSize, shift)
    '''
    # predict and test
    predict = []
    for data in testx:
        predict.append(model.predict(data))
    '''


    '''
    size, window = 10000, 15
    dimIn, dimOut = 60, 20
    trainSet = build_train_set(trainx, size, window, dimIn)
    trainResult = build_train_result(trainy, size, window, dimOut)

    model = km.Sequential()
    model.add(kl.LSTM(20, input_shape=(window,dimIn), activation='sigmoid'))
    model.add(kl.Dense(dimOut))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainSet, trainResult, epochs=10, batch_size=30, verbose=2)
    '''

if __name__ == '__main__':
    main()

