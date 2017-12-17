import sys
import getopt
import os
import xml.etree.ElementTree as et
from sklearn.metrics import f1_score
import numpy as np

ENTITY_DIC = {'ADE':'A',
              'Indication':'I',
              'SSLIF':'S',
              'Severity':'V',
              'Drug':'D', 
              'Dose':'O',
              'Route':'R',
              'Frequency':'F',
              'Duration':'U'}
TAG_ENTITY = 'infon'
TAG_LOC = 'location'
TAG_TEXT = 'text'
ATTR_OFF = 'offset'
ATTR_LEN = 'length'
ENTITY_INDEX = {'A':0, 'I':1, 'S':2, 'V':3, 'D':4, 'O':5, 'R':6, 'F':7, 'U':8, 'X':9}

def append_files(path):
    flist = os.listdir(path)
    newFile = 'joint_file.txt'
    string = ''
    with open(newFile, 'wt') as tar:
        for f in flist:
            with open(os.path.join(path, f), 'rt') as src:
                for line in src:
                    if line[:5] != '19 19':
                        string += line
            tar.write(string)
            string = ''
    return newFile

def merge_entities(fileName):
    string = ''
    with open(fileName, 'rt') as src:
        for line in src:
            content = line.strip('\n').split(' ')
            string = '%s%d %d\n'%(string, int(int(content[0])/2), int(int(content[1])/2))

    newFile = 'merged_file.txt'
    with open(newFile, 'wt') as tar:
        tar.write(string)
    return newFile

def f1_eval(fileName, labels):
    tru = []
    pre = []
    with open(fileName,'rt') as src:
        for line in src:
            content = line.strip('\n').split(' ')
            #if content[0] != '19' or content[1] != '19':
            tru.append(int(content[0]))
            pre.append(int(content[1]))
    print('f1 score for each label:\n', f1_score(tru, pre, labels=labels, average=None))
    print('\nmicro f1 score:', f1_score(tru, pre, labels=labels, average='micro'))
    print('macro f1 score:', f1_score(tru, pre, labels=labels, average='macro'))

# strict or relaxed f1 score
def all_f1_score(predPath='__data__/MADE-1.0/pred', truthPath='__data__/MADE-1.0/annotations'):
    # 9 entities, and 3 scores: TP, FP, and FN
    scores = np.zeros([9,3])
    flist = os.listdir(predPath)
    for f in flist:
        scores += f1_ner(os.path.join(predPath, f), os.path.join(truthPath, f+'.bioc.xml'))
    # compute f1 scores for all entities
    f1_scores = []
    for i in range(scores.shape[0]):
        prec = scores[i,0]/(scores[i,0]+scores[i,1])
        rec = scores[i,0]/(scores[i,0]+scores[i,2])
        f1_scores.append(2*prec*rec/(prec+rec))
    return f1_scores
    
# strict or relaxed f1 score for one file
def f1_ner(predFile, truthFile, strict=False, xpath = './document/passage/annotation'):
    # 9 entities, and three scores for each entity: TP, FP, and FN
    scores = np.zeros([9,3])
    # load prediction
    with open(predFile) as src:
        pred = src.read()
    # avoid out of index
    pred = 'X'+pred+'X'
    pred = list(pred)
    # parse xml truth
    tree = et.parse(truthFile)
    annoList = tree.getroot().findall(xpath)
    # count TP, FP, FN for all entities
    for a in annoList:
        # entity and its symbol
        entity = a.find(TAG_ENTITY).text
        # position
        loc = a.find(TAG_LOC)
        offset = int(loc.get(ATTR_OFF))+1 # +1 due to pred = 'X' + pred + 'X'
        length = int(loc.get(ATTR_LEN))
        if entity == 'PHI':
            continue
        # the corresponding symbol of an entity
        e = ENTITY_DIC[entity]
        result = eval_entity(pred, e, offset, length, strict)
        if result != 0:
            print(entity, offset, length, pred[offset:offset+length])
        del_entity(pred, e, offset, length)
        scores[ENTITY_INDEX[e], result] += 1
    scores += find_FP(pred)
    return scores

# return 0: TP, 1:FP, 2:FN
def eval_entity(pred, entity, offset, length, strict=False):
    end = offset + length
    flag1, flag2, flag3 = False, False, False
    for i in range(offset, end):
        if pred[i].upper() == entity:
            flag1 = True
            break
    for i in range(offset, end):
        if pred[i].upper() != entity:
            flag2 = True
            break
    if pred[offset-1].upper() != entity and pred[end].upper() != entity:
        flag3 = True

    if flag1:
        if (flag3 and not flag2) or not strict:
            return 0
        else:
            return 1
    else:
        return 2

# write "X" at corresponding positions
def del_entity(pred, entity, offset, length):
    end = offset + length
    for i in range(offset, end):
        if pred[i].upper() == entity:
            pred[i] = 'X'
    index = offset-1
    while pred[index].upper() == entity:
        pred[index] = 'X'
        index -= 1
    index = end
    while pred[index].upper() == entity:
        pred[index] = 'X'
        index += 1
    return pred

# find remained FP for all entities
def find_FP(pred):
    result = np.zeros([9,3])
    flag = True
    s = 'X'
    for e in pred:
        sym = e.upper()
        if sym != 'X':
            if flag or sym != s:
                result[ENTITY_INDEX[sym],1] += 1
                flag = False
                s = sym
        else:
            flag = True
    return result

def tests():
    pred = 'XxxaaaxxvvxxrrrarrrX'# a:3, b:8, c1:12, c2:16
    pred = list(pred)
    #entity = 'A'
    #offset, length = 3, 3
    # eval_entity(pred, entity, offset, length, strict=False)
    assert(eval_entity(pred,'A',3,2,strict=True)==1)
    assert(eval_entity(pred,'A',3,4,strict=True)==1)
    assert(eval_entity(pred,'A',3,3,strict=True)==0)
    assert(eval_entity(pred,'A',6,4,strict=True)==2)
    assert(eval_entity(pred,'A',3,2)==0)
    assert(eval_entity(pred,'A',3,3)==0)
    assert(eval_entity(pred,'A',3,4)==0)
    assert(eval_entity(pred,'A',6,4)==2)
    # del_entity(pred, entity, offset, length)
    predCopy = pred[:]
    assert(''.join(del_entity(predCopy, 'R', 14, 1))=='XxxaaaxxvvxxXXXarrrX')
    predCopy = pred[:]
    assert(''.join(del_entity(predCopy, 'R', 14, 2))=='XxxaaaxxvvxxXXXaXXXX')
    # find_FP(pred)
    r = find_FP(list(pred))

def main():
    fileName = '__out__/result_timedis_bi_20e.txt'
    if len(sys.argv) > 1:
        fileName = sys.argv[1]
    labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    labels = [0,1,2,3,4,5,6,7,8]

    options,args = getopt.getopt(sys.argv[1:],"f:d:b:l:m:e:")
    for opt, para in options:
        if opt == '-f':
            fileName = para
    
    f1_eval(fileName, labels)

if __name__ == '__main__':
    tests()
    #main()
