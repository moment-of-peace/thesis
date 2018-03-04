import os
import my_utils as util
import keras.models as km
import gen_dataset as gen
import numpy as np
import pickle

entityIndex= {0:'B-ADE',
              1:'I-ADE',
              2:'B-Indication',
              3:'I-Indication',
              4:'B-SSLIF',
              5:'I-SSLIF',
              6:'B-Severity',
              7:'I-Severity',
              8:'B-Drugname',
              9:'I-Drugname',
              10:'B-Dosage',
              11:'I-Dosage',
              12:'B-Route',
              13:'I-Route',
              14:'B-Frequency',
              15:'I-Frequency',
              16:'B-Duration',
              17:'I-Duration',
              18:'O'}
entityDict = {'A':'B-ADE',
              'a':'I-ADE',
              'I':'B-Indication',
              'i':'I-Indication',
              'S':'B-SSLIF',
              's':'I-SSLIF',
              'V':'B-Severity',
              'v':'I-Severity',
              'D':'B-Drugname',
              'd':'I-Drugname',
              'O':'B-Dosage',
              'o':'I-Dosage',
              'R':'B-Route',
              'r':'I-Route',
              'F':'B-Frequency',
              'f':'I-Frequency',
              'U':'B-Duration',
              'u':'I-Duration',
              'X':'O'}
              
def gen_conll_output(truthPath, predPath, corpPath='__data__/MADE-1.0/corpus/', outPath='__data__/MADE-1.0/'):
    truthList = os.listdir(truthPath)
    predList = os.listdir(predPath)
    corpList = os.listdir(corpPath)
    assert(len(truthList) == len(predList))
    assert(len(truthList) == len(corpList))
    
    for f in truthList:
        with open(os.path.join(truthPath, f)) as src:
            truth = src.read()
        with open(os.path.join(predPath, f)) as src:
            pred = src.read()
        with open(os.path.join(corpPath, f)) as src:
            corp = src.read()
            
        util.write_path_file(outPath, f, convert_conll(f, truth, pred, corp))
        
# process truth and prediction of a single corpus file
def convert_conll(f, truth, pred, corp):
    corp = ' ' + corp + '  '
    start = 0
    pre = ' '
    result = ''
    for i in range(1, len(corp)-1):
        if (not corp[i].isspace()) and start == 0:
            start = i
        if corp[i].isspace():
            if (not corp[i-1].isspace()) and (not is_stop(corp, i-1)):
                result += '%s valid_text_00000 %d %d %s %s %s\n'%(corp[start:i],start-1,i-1,f,entityDict[truth[start-1]],entityDict[pred[start-1]])
            start = 0
        elif is_stop(corp, i):
            if (not corp[i-1].isspace()) and (not is_stop(corp, i-1)):
                result += '%s valid_text_00000 %d %d %s %s %s\n'%(corp[start:i],start-1,i-1,f,entityDict[truth[start-1]],entityDict[pred[start-1]])
            result += '%s valid_text_00000 %d %d %s %s %s\n'%(corp[i],i,i,f,entityDict[truth[i-1]],entityDict[pred[i-1]])
            start = 0
    return result
    
# is stop of a token/phrase/sentence
def is_stop(corp, i):
    if corp[i].isalpha() or corp[i].isdigit():
        return False
    if corp[i] in ['.',','] and corp[i-1].isdigit() and corp[i+1].isdigit():
        return False
    if corp[i] == '.':
        if corp[i-1].isalpha() and i > 1 and corp[i-2] == '.':
            return False
        if corp[i-1].isalpha() and corp[i+1].isalpha():
            return False
    if corp[i] == '/':
        if not(corp[i-1].isspace() or corp[i+1].isspace()):
            return False
    
    return True

# test
def test_conll():
    f = '1_1'
    truth = 'AAXIIiiiiXXSSXDDDddddddXAAAAAXddddddXsXXIIIIXXXAAAAX'
    pred = truth
    corp = 'ab dc kjl  12 2.3 1,000 hh/mm 4+55 h,s. a.m. . yu3x.'
    result = convert_conll(f, truth, pred, corp)
    print(corp)
    print(result)

def cmp_file(x):
    c = x.split('_')
    return int(c[0]) * 100000 + int(c[1])
    
# get the prediction and truth
def gen_pred_tru(modelFile, shiftSize, shift, multi):
    if multi:
        trainPath = '__data__/MADE-1.0/process2_stepFour_corp'
        truthPath = '__data__/MADE-1.0/process2_stepThree_entity'
    else:
        trainPath = '__data__/MADE-1.0/process_stepFour_corp'
        truthPath = '__data__/MADE-1.0/process_stepThree_entity'
    # cross validation
    flist = util.sorted_file_list(trainPath, cmp_file)
    flist = gen.cycleShift(flist, shift, shiftSize)
    vocabPath = 'vocab_made_8000.pkl'
    with open(vocabPath, 'rb') as handle:   # load vocabulary from file
        vocab = pickle.load(handle)
    # generate x and y
    trainx = gen.getIndex(trainPath, flist, vocab)
    if multi:
        trainy = gen.genResultBin(truthPath, flist)
    else:
        trainy = gen.genResult(truthPath,flist)
    '''
    for i in range(len(trainy)):
        for j in range(len(trainy[i])):
            try:
                index = trainy[i][j].index(1)
            except Exception:
                print(Exception)
                print(flist[i],j)
                exit()
    '''
    testData, testResult = gen.genTrainDataset(trainx[0:shiftSize], trainy[0:shiftSize], 20, 20, multi=multi)
    
    model = km.load_model(modelFile)
    predict = model.predict(np.array(testData))
    return flist[0:shiftSize], predict, testResult
    
# determine whether the token has multiple lables
def is_multi(entity):
    index1 = entity.index(1)
    try:
        index2 = entity.index(1, index1+1)
    except ValueError:
        return False
    return True
    
# to investigate 
def print_multi(predict, truth):
    f = open('multi', 'wt')
    for i in range(1000):
        for j in range(len(truth[i])):
            #if is_multi(truth[i][j]):
            f.write(str(predict[i][j])+'\n')
            f.write(str(truth[i][j])+'\n\n')
    f.close()

if __name__ == '__main__':
    flist, pre, tru = gen_pred_tru('model_made_90-0_36-epoch.h5', 90, 0)
    print_multi(pre,tru)
    '''
    corpPath = '__data__/MADE-1.0/process_stepFour_corp'
    with open('con_output', 'wt') as tar:
        i = 0
        for f in flist:
            spaces = []
            with open(os.path.join(corpPath, f)) as src:
                content = src.read()
                # find positions of spaces
                for m in range(len(content)):
                    if content[m] == ' ':
                        spaces.append(m)
                text = content.split(' ')
            length = len(text)
            j = 0
            start = 0
            while True:
                for k in range(20):
                    #index = j*20 + k
                    predict = entityIndex[util.maxIndex(pre[i][k])]
                    truth = entityIndex[tru[i][k].index(1)]
                    word = text[j]
                    tar.write('%s valid_text_00000 %d %d %s %s %s\n'%(word, start, start+len(word), f, truth, predict))
                    if text[j] == '.':
                        tar.write('\n')
                    j += 1
                    if (j) >= length:
                        break
                    start = spaces[j-1]+1
                i += 1
                if j >= length:
                    break
            if text[-1] != '.':
                tar.write('\n')
    '''