'''
generate corresponding results for training documents
Author: Yi Liu
'''
import numpy as np
import keras.layers as kl
import keras.models as km

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


# generate a taining dataset from a directory
def gen_train_data(path, filelist):
    trainSet = []

    for f in filelist:
        trainSet.append(embed_file(path+f, 'e'))
    return trainSet





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

embedModel = np.load('weights_glove100.npy')
data, result = np.load('tx.npy'), np.load('ty.npy')
trainData, trainResult = data[:-600], result[:-600]
testData, testResult = data[-600:], result[-600:]
del data
del result
# build and fit model
model = km.Sequential()
model.add(kl.Embedding(embedModel.shape[0],embedModel.shape[1], mask_zero=True,weights=[embedModel]))
model.add(kl.Bidirectional(kl.LSTM(20,activation='relu',return_sequences=True)))
model.add(kl.Bidirectional(kl.LSTM(20, return_sequences=True)))
model.add(kl.TimeDistributed(kl.Dense(20)))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainData, trainResult, epochs=5, batch_size=100, verbose=2)
# predict and evaluate
predict = model.predict(testData)
n = 0
shape = predict.shape
for i in range(shape[0]):
    for j in range(shape[1]):
        if maxIndex(predict[i][j]) == maxIndex(testResult[i][j]):
            n += 1
print('ave accurate: %f'%(n/shape[0]/shape[1]))
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

