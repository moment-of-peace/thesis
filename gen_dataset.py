import os

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
        data.extend(window(trainx[i],windowsize,step,0)) # 0 or 1 ?
        result.extend(window(trainy[i],windowsize,step,[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))
    return data, result 
def window(data, windowsize, step, padding):
    result = []
    for i in range(0, len(data)-windowsize+1, step):
        result.append(data[i:i+windowsize])
    remainder = (len(data)-windowsize)%step
    if remainder != 0:
        last = data[-remainder:]
        for i in range(remainder, windowsize):
            last.append(padding)
        result.append(last)
    return result

# do a cycling shift on a list
def cycleShift(flist, shift, shiftSize):
    assert(shift*shiftSize < len(flist))
    shiftedList = flist[shift*shiftSize:]
    shiftedList.extend(flist[0:shift*shiftSize])
    return shiftedList

