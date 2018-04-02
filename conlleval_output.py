import os
import sys
import my_utils as util
import keras.models as km
from keras_contrib.layers.crf import CRF
import gen_dataset as gen
import numpy as np
import pickle

OTHER = 'O'
ENTITIES = [OTHER,'B-ADE','I-ADE','B-Indication','I-Indication','B-SSLIF','I-SSLIF','B-Severity',\
'I-Severity','B-Drugname','I-Drugname','B-Dosage','I-Dosage','B-Route','I-Route','B-Frequency',\
'I-Frequency','B-Duration','I-Duration']
SYMBOLES = ['X','A','a','I','i','S','s','V','v','D','d','O','o','R','r','F','f','U','u']

entityIndex = {i:ENTITIES[i] for i in range(19)}

entityDict = {SYMBOLES[i]:ENTITIES[i] for i in range(19)}
              
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
def gen_pred_tru(modelFile, shiftSize, shift, multi, sent=False, nclass=0):
    if multi:
        trainPath = '__data__/MADE-1.0/process2_stepFour_corp%d'%(shift)
        truthPath = '__data__/MADE-1.0/process2_stepThree_entity%d'%(shift)
    else:
        trainPath = '__data__/MADE-1.0/process_stepFour_corp'
        truthPath = '__data__/MADE-1.0/process_stepThree_entity'
    # cross validation
    flist = util.sorted_file_list(trainPath, cmp_file)
    #flist = gen.cycleShift(flist, shift, shiftSize)
    vocabPath = 'vocab_made_8000.pkl'
    with open(vocabPath, 'rb') as handle:   # load vocabulary from file
        vocab = pickle.load(handle)
    # generate x and y
    x = gen.getIndex(trainPath, flist[:shiftSize], vocab)
    if multi:
        y = gen.genResultBin(truthPath, flist[:shiftSize])
    else:
        y = gen.genResult(truthPath,flist[:shiftSize])
    if nclass != 0:
        y = gen.twoClass(y, nclass)
    if sent:
        testx, testy =  gen.toSent(flist[:shiftSize], trainPath, x, y)
        testData, testResult = gen.padSent(testx, testy, 100)
    else:
        testData, testResult = gen.genTrainDataset(x, y, 20, 20)
    
    model = km.load_model(modelFile)
    predict = model.predict(np.array(testData))
    return flist[0:shiftSize], predict, testResult
    
def gen_pred_tru_test(modelFile, multi, sent=False, nclass=0):
    if multi:
        trainPath = '__data__/MADE-1.0-test/process2_stepFour_corp'
        truthPath = '__data__/MADE-1.0-test/process2_stepThree_entity'
    else:
        trainPath = '__data__/MADE-1.0-test/process_stepFour_corp'
        truthPath = '__data__/MADE-1.0-test/process_stepThree_entity'
    flist = os.listdir(trainPath)
    #flist = gen.cycleShift(flist, shift, shiftSize)
    vocabPath = 'vocab_made_8000.pkl'
    with open(vocabPath, 'rb') as handle:   # load vocabulary from file
        vocab = pickle.load(handle)
    # generate x and y
    x = gen.getIndex(trainPath, flist, vocab)
    if multi:
        y = gen.genResultBin(truthPath, flist)
    else:
        y = gen.genResult(truthPath,flist)
    if nclass != 0:
        y = gen.twoClass(y, nclass)
    if sent:
        testx, testy =  gen.toSent(flist, trainPath, x, y)
        testData, testResult = gen.padSent(testx, testy, 100)
    else:
        testData, testResult = gen.genTrainDataset(x, y, 20, 20)
    
    model = km.load_model(modelFile)
    predict = model.predict(np.array(testData))
    return flist, predict, testResult
    
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

# process a token
def conll_formatter(word, start, fname, pred, tru, coeff):
    prediction = util.findIndex(pred, coeff=coeff)
    truth = util.findIndex(tru, thres=1)
    if len(prediction) == 1 and len(truth) == 1:
        # the token only has one label
        return '%s valid_text_00000 %d %d %s %s %s\n'%(word, start, start+len(word), fname, \
entityIndex[truth[0]], entityIndex[prediction[0]])

    result = ''
    inf = float('inf')
    prediction.append(inf)
    truth.append(inf)
    i, j = iter(prediction), iter(truth)
    p, t = next(i), next(j)
    while p != inf or t != inf:
        if p == t:
            result += '%s valid_text_00000 %d %d %s %s %s\n'%(word, start, start+len(word), fname, entityIndex[t], entityIndex[t])
            p, t = next(i), next(j)
        elif p < t:
            result += '%s valid_text_00000 %d %d %s %s %s\n'%(word, start, start+len(word), fname, OTHER, entityIndex[p])
            p = next(i)
        else:
            result += '%s valid_text_00000 %d %d %s %s %s\n'%(word, start, start+len(word), fname, entityIndex[t], OTHER)
            t = next(j)
    return result

# generate a conll style output file
def conll_output(flist, corpPath, pred, tru, coeff):
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
                    tar.write(conll_formatter(text[j], start, f, pred[i][k], tru[i][k], coeff))
                    if text[j] == '.':
                        tar.write('\n')
                    j += 1
                    if j >= length:
                        break
                    start = spaces[j-1]+1
                i += 1
                if j >= length:
                    break
            if text[-1] != '.':
                tar.write('\n')
def conll_output_s(flist, corpPath, pred, tru, coeff, index=0):
    with open('con_output%d'%(index), 'wt') as tar:
        i = 0
        for f in flist:
            spaces = [-1]
            with open(os.path.join(corpPath, f)) as src:
                content = src.read().strip()
                # find positions of spaces
                for m in range(len(content)):
                    if content[m] == ' ':
                        spaces.append(m)
                text = content.split('\n')
            for sent in text:
                words = sent.strip().split(' ')
                
                for j in range(len(words)):
                    start = spaces[j]+1
                    tar.write(conll_formatter(words[j], start, f, pred[i][j], tru[i][j], coeff))
                i += 1
                

if __name__ == '__main__':
    P = '__data__/MADE-1.0-test/'
    multi = True
    coeff = float(sys.argv[3])
    modelFile = 'model_made_36-epoch.h5'
    #modelFile = 'models-scikit-10/model_made_90-%s_%s-epoch.h5'%(sys.argv[2], sys.argv[1])
    print(modelFile, coeff)
    if sys.argv[4] == 'w':# word level
        corpPath = P+'process2_stepFour_corp'+sys.argv[2] 
        flist, pred, tru = gen_pred_tru(modelFile, 90, int(sys.argv[2]), multi, nclass=int(sys.argv[5]))
        #print_multi(pred,tru)
        conll_output(flist, corpPath, pred, tru, coeff)
    elif sys.argv[4] == 's':# sentence level
        corpPath = P+'corp_sent_cut100'
        flist, pred, tru = gen_pred_tru_test(modelFile, multi, sent=True, nclass=int(sys.argv[5]))
        conll_output_s(flist, corpPath, pred, tru, coeff)
        '''
        flist, pred, tru = gen_pred_tru(modelFile, 90, int(sys.argv[2]), multi, sent=True, nclass=int(sys.argv[5]))
        #print_multi(pred,tru)
        flist = gen.cycleShift(util.sorted_file_list(corpPath, cmp_file), int(sys.argv[2]), 90)
        conll_output_s(flist[:90], corpPath, pred, tru, coeff)
        '''