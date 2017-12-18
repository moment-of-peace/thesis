import os
import pickle
import numpy as np
import my_utils as util

'''
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
'''
entityDict = {'A':[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              'a':[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              'I':[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              'i':[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              'S':[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              's':[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              'V':[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              'v':[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              'D':[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              'd':[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              'O':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              'o':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
              'R':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              'r':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              'F':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
              'f':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
              'U':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
              'u':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
              'X':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}

# reversedDict = {0:'a', 1:'i', 2:'s', 3:'v', 4:'d', 5:'o', 6:'r', 7:'f', 8:'u', 9:'X'}
reversedDict = {0:'A',1:'a',2:'I',3:'i',4:'S',5:'s',6:'V',7:'v',8:'D',9:'d',10:'O',11:'o',12:'R',13:'r',14:'F',15:'f',16:'U',17:'u',18:'X'}
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
    results = []
    for f in flist:
        entities = []
        with open(os.path.join(path, f), 'rt') as src:
            content = src.read().split(' ')
        for word in content:
            try:
                entities.append(entityDict[word[0]])
            except:
                print(f, word)
                exit()
            #entities.append(entityDict[word[0].lower()])
        results.append(entities)
    return results

# use window method to cut datasets for training
def genTrainDataset(trainx, trainy, windowsize, step):
    data = []
    result = []
    for i in range(0, len(trainx)):
        data.extend(window(trainx[i],windowsize,step,0)) # 0 or 1 ?
        result.extend(window(trainy[i],windowsize,step,entityDict['X']))
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

def restore(flist, predict, windowSize=20, path='__data__/MADE-1.0/', newPath='__data__/MADE-1.0/restore'):
    if not os.path.exists(newPath):
        os.makedirs(newPath)

    index = 0
    for f in flist:
        with open(os.path.join(path,'process_stepThree_entity',f)) as src:
            truth = src.read().split(' ')
        length = len(truth)
        entity = ''
        for i in range(length):
            entity += duplicate(reversedDict[predict[index]], len(truth[i])) + ' '
            index += 1
        # three steps of restoring
        entity = entity[:-1]
        
        entity = reverseStepOne(entity, os.path.join(path, 'process_stepThree_trace'), f)
        entity = reverseStepTwo(entity, os.path.join(path, 'process_stepTwo_trace'), f)
        entity = reverseStepThree(entity, os.path.join(path, 'process_stepOne_trace'), f)
        
        with open(os.path.join(newPath, f), 'wt') as tar:
            tar.write(entity)

        if index%windowSize != 0:
            index = windowSize * (int(index/windowSize)+1)

def duplicate(chara, num):
    result = ''
    for i in range(0, num):
        result += chara
    return result

def readTrace(tracePath, fileName):
    with open(os.path.join(tracePath,fileName)) as src:
        trace = src.read().strip('\n')
    if trace == '':
        return ''
    else:
        return trace.split('\n')

# corresponds to step three in preprocessing, add spaces
def reverseStepOne(entity, tracePath, fileName):
    trace = readTrace(tracePath, fileName)
    for t in trace:
        tr = t[1:-1].split(', ')
        pos, num = int(tr[0]), int(tr[1])
        entity = entity[:pos] + duplicate(' ', -num) + entity[pos:]
        #print(entity, len(duplicate(' ', -num)))
    return entity

# corresponds to step two in preprocessing, remove spaces
def reverseStepTwo(entity, tracePath, fileName):
    trace = readTrace(tracePath, fileName)
    for t in trace:
        tr = t[1:-1].split(', ')
        pos = int(tr[0])
        entity = entity[:pos] + entity[pos+1:]
    return entity

# corresponds to step one in preprocessing, add signals, and convert spaces
def reverseStepThree(entity, tracePath, fileName):
    trace = readTrace(tracePath, fileName)
    #entity = ' ' + entity
    for t in trace:
        tr = t[1:-1].split(', ')
        pos, num = int(tr[0]), int(tr[1])
        try:
            entity = entity[:pos] + duplicate(entity[pos-1], -num) + entity[pos:]
        except:
            print(fileName, t, len(entity))
            exit()
    # restore spaces to entities
    if entity[0].isspace(): # deal with entity[0] separately, to avoid out of index in while loop
        newEntity = 'X'
    else:
        newEntity = entity[0]
    i = 1
    while i < len(entity):
        
        if entity[i].isspace():
            t = countSpace(entity, i)
            if t[1].isupper() or t[1].lower() != entity[i-1].lower():
                newEntity += duplicate('X', t[0])
            else:
                newEntity += duplicate(t[1], t[0])
                
            i += t[0]
        else:
            newEntity += entity[i]
            i += 1
    return newEntity

# count how many consistent spaces
def countSpace(entity, index):
    num = 0
    while index < len(entity) and entity[index].isspace():
        num += 1
        index += 1
    if index < len(entity):
        return (num, entity[index])
    else:
        return (num, 'X')

def checkRestore(path1='__data__/MADE-1.0/entities', path2='__data__/MADE-1.0/restore'):
    flist = os.listdir(path1)
    flist2 = os.listdir(path2)
    assert(len(flist) == 876)
    assert(len(flist2) == 876)
    
    string = ''
    for f in flist:
        #print(f)
        with open(os.path.join(path1, f), 'rt') as src:
            raw = src.read()
        with open(os.path.join(path2, f), 'rt') as src:
            restore = src.read()
        with open('__data__/MADE-1.0/process_stepThree_corp/'+f) as src:
            corp = src.read()
        #assert(len(raw) == len(restore))
        if len(raw) != len(restore):
            print(f, 'length', len(raw), len(restore))
            exit()
        
        for i in range(len(raw)):
            if raw[i].lower() != restore[i].lower():
                string += '%s %d\n%s\n%s\n%s\n\n'%(f, i, raw[i-15:i+15], restore[i-15:i+15], util.del_linefeed(corp[i-15:i+15]))
    if string != '':
        with open('wrongs','wt') as fid:
            fid.write(string)
        print('wrongs')
    print('checks finished')

def checkAll():
    windowSize = 20
    vocabPath = 'vocab_made_8000.pkl'
    path = '__data__/MADE-1.0/'
    newPath = '__data__/MADE-1.0/restore'
    trainPath = '__data__/MADE-1.0/process_stepFour_corp'
    truthPath = '__data__/MADE-1.0/process_stepThree_entity'
    '''
    with open(vocabPath, 'rb') as handle:   # load vocabulary from file
        vocab = pickle.load(handle)
    flist = os.listdir(truthPath)
    trainx = getIndex(trainPath, flist, vocab)
    trainy = genResult(truthPath,flist)
    testData, testResult = genTrainDataset(trainx, trainy, windowSize, windowSize)
    result = []
    for l in testResult:
        for e in l:
            result.append(util.maxIndex(np.array(e)))
    restore(flist, result, windowSize=20, path='__data__/MADE-1.0/', newPath='__data__/MADE-1.0/restore')
    '''
    checkRestore(os.path.join(path, 'process_stepThree_entity'), newPath)

def main():
    path = '__data__/MADE-1.0/'
    newPath = '__data__/MADE-1.0/restore'
    if not os.path.exists(newPath):
        os.makedirs(newPath)

    entityPath = 'restore-2'
    #entityPath = 'process_stepThree_entity'
    flist = os.listdir(os.path.join(path,entityPath))
    for f in flist:
        with open(os.path.join(path,entityPath,f)) as src:
            entity = src.read()
        
        # three steps of restoring
        entity = reverseStepOne(entity, os.path.join(path, 'process_stepThree_trace'), f)
        entity = reverseStepTwo(entity, os.path.join(path, 'process_stepTwo_trace'), f)
        entity = reverseStepThree(entity, os.path.join(path, 'process_stepOne_trace'), f)
        with open(os.path.join(newPath, f), 'wt') as tar:
            tar.write(entity)
    
    checkRestore(os.path.join(path, 'entities'), newPath)

def cmp_file(x):
    c = x.split('_')
    return int(c[0]) * 100000 + int(c[1])

if __name__ == '__main__':
    #checkAll()
    #main()
    path = '__data__/MADE-1.0/corpus'
    flist = util.sorted_file_list(path, cmp_file)
    results = []
    with open('joint.txt') as f:
        for line in f:
            results.append(int(line.strip('\n').split(' ')[1]))
    restore(flist, results, windowSize=20, path='__data__/MADE-1.0/', newPath='__data__/MADE-1.0/pred1218')
            
    '''
    path = '__data__/MADE-1.0/'
    newPath = '__data__/MADE-1.0/restore'
    if not os.path.exists(newPath):
        os.makedirs(newPath)
    
    flist = os.listdir(os.path.join(path,'process_stepThree_entity'))
    for f in flist:
        with open(os.path.join(path,'process_stepThree_entity',f)) as src:
            entity = src.read()
        
        # three steps of restoring
        entity = reverseStepOne(entity, os.path.join(path, 'process_stepThree_trace'), f)
        #entity = reverseStepTwo(entity, os.path.join(path, 'process_stepTwo_trace'), f)
        #entity = reverseStepThree(entity, os.path.join(path, 'process_stepOne_trace'), f)
        with open(os.path.join(newPath, f), 'wt') as tar:
            tar.write(entity)
    
    checkRestore(os.path.join(path, 'process_stepTwo_entity'), newPath)
    '''
