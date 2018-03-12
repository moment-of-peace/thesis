import os
import numpy as np
import pickle

def write_list(l, fileName=None, linefeed='\n'):
    string = ''
    for e in l:
        string = string + str(e) + linefeed
    if fileName != None:
        with open(fileName,'wt') as tar:
            tar.write(string)
    return string
            
def write_dict(d, fileName=None, linefeed='\n'):
    string = ''
    for k in d.keys():
        string = '%s%s,%s%s'%(string, str(k), str(d[k]), linefeed)
    if fileName != None:
        with open(fileName,'wt') as tar:
            tar.write(string)
    return string
    
# return a sorted list of files in a directory
def sorted_file_list(path, comparator):
    flist = os.listdir(path)
    return sorted(flist, key=comparator)

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

# find multiple index 
def findIndex(array, coeff=1, thres=None):
    indexs = []
    if thres == None:
        thres = np.amax(array) * coeff
    for i in range(len(array)):
        if array[i] >= thres:
            indexs.append(i)
    if (0 in indexs) and (len(indexs) > 1):
        indexs.remove(0)
    return indexs
    
# write a file into a directory, create the path if not exists
def write_path_file(path, fileName, content, flag='wt'):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, fileName), flag) as tar:
        tar.write(content)
        
# do some processing for each file in a directory
def process_files(path, func):
    flist = os.listdir(path)
    for f in flist:
        with open(os.path.join(path, f), 'rt') as src:
            content = src.read()
        func(content)
        
# join all files in a directory and form a single file
def join_files(path, fileName='joint_file.txt', separator=''):
    flist = os.listdir(path)
    sep = ''
    for f in flist:
        print(f)
        with open(os.path.join(path, f), 'rt') as src:
            content = src.read()
        with open(fileName, 'at') as tar:
            tar.write(sep + content)
        sep = separator

# generate vocabulary (a dict) and weights (2D np array) from a gensim style embedding file
def gen_weights_vocab(modelFile, name='', saveFlag=True):
    vocab = {}  # dict {'word': index, ...}
    with open(modelFile,'r') as f:
        line = f.readline()
        [length, dim] = line.split(' ')
        weights = np.zeros((int(length)+1, int(dim)), dtype = np.float64)    #  2-d array [[vector], ...]
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
            weights[i] = np.array(vector)
            line = f.readline()
            i = i+1
    if saveFlag:
        # write into files
        np.save('weights_%s_%d.npy'%(name, len(vocab)), weights)
        with open('vocab_%s_%d.pkl'%(name, len(vocab)), 'wb') as handle:
            pickle.dump(vocab, handle)
    return vocab, weights

# replace line feed
def del_linefeed(string, chara=' '):
    result = ''
    for c in string:
        if c == '\n':
            result += chara
        else:
            result += c
    return result

# convert a number to a list of 0 and 1 which indicates the binary format
# example: 6 -> [0, 1, 1]
def numToBin(num):
    result = []
    for i in range(19):
        result.append(num&1)
        num = num >> 1
    return result
    
if __name__ == '__main__':
    join_files('__data__/MADE-1.0/process_stepFour_corp', separator=' . ')
    '''
    gen_weights_vocab('__data__/word2vec_model_made_6000.txt', name='_made_6000')
    gen_weights_vocab('__data__/word2vec_model_made_8000.txt', name='_made_8000')
    gen_weights_vocab('__data__/word2vec_model_made_10000.txt', name='_made_10000')
    '''
