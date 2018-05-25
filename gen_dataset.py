import os
import pickle
import numpy as np
import my_utils as util


def gen_entity_vec(index):
    return [1 if i==index else 0 for i in range(19)]
    
entities = ['X','A','a','I','i','S','s','V','v','D','d','O','o','R','r','F','f','U','u']
entityDict = {entities[i]:gen_entity_vec(i) for i in range(19)}
reversedDict = {i:entities[i] for i in range(19)}

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

# return a list ("results") of lists, and each nested list ("entities", represents a file) is a list of entity vectors
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
def genResultBin(path, flist):
    results = []
    for f in flist:
        entities = []
        c = np.load(os.path.join(path, f+'.npy'))
        entities.append(util.numToBin(c[0]))
        for i in range(1,len(c)):
            if c[i] == 0: #split spaces
                entities.append(util.numToBin(c[i+1]))
        results.append(entities)
    return results

# use window method to cut datasets for training
def genTrainDataset(trainx, trainy, windowsize, step,padding=None):
    data = []
    result = []
    for i in range(0, len(trainx)):
        data.extend(window(trainx[i],windowsize,step,1)) # 0 or 1 ?
        if padding == None:
            result.extend(window(trainy[i],windowsize,step,entityDict['X']))
        else:
            result.extend(window(trainy[i],windowsize,step,padding))
            
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
    '''
    assert(shift*shiftSize < len(flist))
    shiftedList = flist[shift*shiftSize:]
    shiftedList.extend(flist[0:shift*shiftSize])
    return shiftedList
    '''
    assert(shift*shiftSize < len(flist))
    start = shift*shiftSize
    shiftedList = flist[start:min(start+shiftSize, len(flist))]
    shiftedList.extend(flist[0:start])
    shiftedList.extend(flist[start+shiftSize:len(flist)])
    return shiftedList
    

# split tain data by sentences
# input: trainx: 2-d, trainy: 3-d; output: x: 2-d, y: 3-d
def toSent(flist, corpPath, trainx, trainy):
    x, y = [], []
    for j in range(len(flist)):
        newText = ''
        with open(os.path.join(corpPath, flist[j])) as src:
            text = src.read().strip().split(' ')
            start = 0
            for i in range(len(text)):
                if text[i] == '.':
                    x.append(trainx[j][start:i+1])
                    y.append(trainy[j][start:i+1])
                    start = i+1
            if text[-1] != '.':
                x.append(trainx[j][start:])
                y.append(trainy[j][start:])
    return x, y

# pad sentences to the same length
def padSent(x, y, length):
    sizey = len(y[0][0])
    x_pad, y_pad = [], []
    for i in range(len(x)):
        sent_len = len(x[i])
        if sent_len > length:
            x_pad.extend(cutSent(x[i], length, 1))
            y_pad.extend(cutSent(y[i], length, entityDict['X'][0:sizey]))
        else:
            tempx, tempy = x[i][:], y[i][:]
            tempx.extend([1 for i in range(length-sent_len)])
            tempy.extend([entityDict['X'][0:sizey] for i in range(length-sent_len)])
            x_pad.append(tempx)
            y_pad.append(tempy)
    return x_pad, y_pad
# cut a long sentences
def cutSent(s, length, pad):    
    s_cut = []
    for i in range(int(len(s)/length)):
        start = i * length
        s_cut.append(s[start:start+length])
    m = len(s)%length
    if m != 0:
        tail = s[-m:]
        tail.extend([pad for i in range(length-m)])
        s_cut.append(tail)
    return s_cut

# generate charcater level indices. input: 2-d list. output: 3-d
def gen_char_indices(token_indices, vocab):
    words = {v:k for k,v in vocab.items()}
    char_indices = []
    for sent in token_indices:
        sent_char = []
        for index in sent:
            word = words[index].lower()
            word_char = [to_char_index(c) for c in word[:10]]
            if len(word_char) < 10:
                word_char.extend([0 for i in range(10-len(word_char))])
            sent_char.append(word_char)
        char_indices.append(sent_char)
    return char_indices
def to_char_index(chara):
    num = 63
    if chara.isalpha():
        num = ord(chara) - ord('a') + 1
    elif chara.isdigit():
        num = 56
    return num  #[(num&2**i)>>i for i in range(5)]

# test toSent
def testToSent():
    def cmp_file(x):
        c = x.split('_')
        return int(c[0]) * 100000 + int(c[1])
        
    flist = util.sorted_file_list(trainPath, cmp_file)
    trainx = getIndex(trainPath, flist, vocab)
    trainy = genResultBin(truthPath, flist)
    print('step 1')
    assert(len(trainx) == 876)
    assert(len(trainy) == 876)
    
    x, y = gen.toSent(flist, trainPath, trainx, trainy)
    print('step 2')
    assert(len(x) == len(y))
    for i in range(len(x)):
        assert(len(x[i]) == len(y[i]))
    assert(len(x[0]) == 15)
    assert(len(x[3]) == 2)
    assert(len(x[99]) == 23)
    assert(len(x[101]) == 5)
    assert(len(x[118]) == 27)
    assert(len(x[123]) == 16)
    
    x_pad, y_pad = padSent(x, y, 100)
    print('step 3')
    assert(len(x_pad) == len(y_pad))
    for i in range(len(x)):
        assert(len(x_pad[i]) == 100)
        assert(len(y_pad[i]) == 100)
    
# convert to two-class
def twoClass(data, n):
    result = []
    for entity in data:
        newEnti = []
        for vec in entity:
            v = [0,0,0]
            v[0] = vec[0]
            v[1] = vec[n[0]]
            v[2] = vec[n[1]]
            newEnti.append(v)
        result.append(newEnti)
    return result

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
    pass
    '''
    path = '__data__/MADE-1.0/corpus'
    flist = util.sorted_file_list(path, cmp_file)
    results = []
    with open('joint.txt') as f:
        for line in f:
            results.append(int(line.strip('\n').split(' ')[1]))
    restore(flist, results, windowSize=20, path='__data__/MADE-1.0/', newPath='__data__/MADE-1.0/pred1218')
    '''    
    