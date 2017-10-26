import numpy as np
import os
import pickle

# load binary vocabulary and embedding weights file
def load_model(modelFile):
    weights = np.load('weights_%s.npy'%(modelFile))  # load weights from file
    with open('vocab_%s.pkl'%(modelFile), 'rb') as handle:   # load vocabulary from file
        vocab = pickle.load(handle)
    return vocab, weights

def loadW2VModel(modelFile):
    return gensim.models.KeyedVectors.load_word2vec_format(modelFile, binary=False)

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
                        vector.append(vocab['unk'])
                    except:
                        vector.append(vocab['a'])
    return vector

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
                    result.append(entityDict[entity])
                    #result.append(entityDict[entity].index(1))
        trainResult.append(result)
    return trainResult

# use window method to cut datasets for training
def gen_train_dataset(trainx, trainy, windowsize, step):
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
    return result

def drop(data, rate):
    result = []
    for i in range(len(data)):
        if i%rate == 0:
            result.append(data[i])
    return result

trainPath = '2009train/'
truthPath = '2009truth/'
modelFile = 'glove100'
'''
loadW2VModel(w2vModel)
flist = commonFiles(trainPath,truthPath)
trainx = gen_train_data(trainPath, flist)
trainy = gen_train_result(trainPath,truthPath,flist)

#en = extract_true_entity('truth/72791')
#match_file('train/72791', en)
maxLen = 0  # the max artical length
for i in range(0, len(trainx)):
    #print(len(trainx[i]))
    if len(trainx[i]) > maxLen:
        maxLen = len(trainx[i])
print('max length: ' + str(maxLen))
[trainx, trainy] = padding(trainx, trainy, maxLen)
'''

vocab, embedModel = load_model(modelFile)
flist = commonFiles(trainPath,truthPath)
trainx = get_index(trainPath, flist, vocab)
trainy = gen_train_result(trainPath,truthPath,flist)
trainData,trainResult = gen_train_dataset(trainx[:220], trainy[:220], 20, 1)
trainData = drop(trainData, 5)
trainResult = drop(trainResult, 5)
testData, testResult = gen_train_dataset(trainx[220:], trainy[220:], 20, 20)
np.save('tx',np.array(trainData))
np.save('ty',np.array(trainResult))
np.save('testx',np.array(testData))
np.save('testy',np.array(testResult))

