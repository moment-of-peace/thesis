import os
import shutil
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
        if thres == 0:
            thres = 1
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
def join_files(path, fileName='joint_file', separator=' ', discard=False):
    flist = os.listdir(path)
    text, sep = '', ''
    for f in flist:
        print(f)
        with open(os.path.join(path, f), 'rt') as src:
            content = src.read().strip()
        if discard:
            # discard some head and tail meaningless text
            start = content.find('clinic note')
            end = content.find(' e - signed')
            if end < 0:
                end = content.find(' signed by')
            start = start if start > 0 else 0
            end = end if end > 0 else len(content)
            text += (sep + content[start+12:end])
        else:
            text += (sep + content)
        sep = separator
    with open(fileName, 'wt') as tar:
        tar.write(text)

# generate vocabulary (a dict) and weights (2D np array) from a gensim style embedding file
def gen_weights_vocab(modelFile, name='', saveFlag=True):
    vocab = {}  # dict {'word': index, ...}
    with open(modelFile,'rb') as f:
        line = f.readline().decode('utf-8')
        [length, dim] = line.split(' ')
        weights = np.zeros((int(length)+1, int(dim)), dtype = np.float64)    #  2-d array [[vector], ...]
        line = f.readline().decode('utf-8')
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
            line = f.readline().decode('utf-8')
            i = i+1
    if 'unk' in vocab:
        vocab['UNK'] = vocab.pop('unk')
    if saveFlag:
        # write into files
        np.save('weights_%s_%d.npy'%(name, len(vocab)), weights)
        with open('vocab_%s_%d.pkl'%(name, len(vocab)), 'wb') as handle:
            pickle.dump(vocab, handle)
    return vocab, weights

# generate weights for char level embedding
def gen_char_weights(dim, path='', saveFlag=False):
    weights = []
    for i in range(2**dim):
        weights.append([(i&2**j)>>j for j in range(dim)])
    weights = np.array(weights)
    if saveFlag:
        np.save('%schar_weights_%d'%(path, dim), weights)
    return weights

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
def numToBin(num, length=19):
    result = []
    for i in range(length):
        result.append(num&1)
        num = num >> 1
    return result

def shiftFiles(flist, path, newPath, shiftSize, shift, tail=''):    
    if not os.path.exists(newPath):
        os.makedirs(newPath)
        
    start = shiftSize*shift
    for i in range(0, start):
        shutil.copy(os.path.join(path,flist[i]+tail),os.path.join(newPath,flist[i]+tail))
        
    for i in range(start, min(start+shiftSize, len(flist))):
        shutil.copy(os.path.join(path,flist[i]+tail),os.path.join(newPath,'0_%d%s'%(i,tail)))
    for i in range(start+shiftSize, len(flist)):
        shutil.copy(os.path.join(path,flist[i]+tail),os.path.join(newPath,flist[i]+tail))
    
if __name__ == '__main__':
    #join_files('__data__/MADE2-1.0/process2_stepFour_corp', separator=' . ',discard=False)
    
    gen_weights_vocab('word2vec_model_withdiscard.txt', name='discard')
    #gen_weights_vocab('__data__/word2vec_model_made_8000.txt', name='_made_8000')
    #gen_weights_vocab('__data__/word2vec_model_made_10000.txt', name='_made_10000')
    
