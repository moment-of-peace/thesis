import os
import numpy as np
import pickle
import keras.models as km
from xml.etree import ElementTree as et
import gen_dataset as gen
import my_utils as util

RAWENTITY=['ADE','Indication','SSLIF','Severity','Drug','Dose','Route','Frequency','Duration']
global anno_id

def writeXML(fileName, corpus, index, entities, coeff=1, path='__data__/MADE-1.0-test/xml/',nclass=0):
    global anno_id
    root = et.Element('collection')
    source = et.SubElement(root, 'source')
    date = et.SubElement(root, 'data')
    key = et.SubElement(root, 'key')

    doc = et.SubElement(root, 'document')
    docid = et.SubElement(doc, 'id')
    docid.text = fileName

    passage = et.SubElement(doc, 'passage')
    offset = et.SubElement(passage, 'offset')
    offset.text = '0'
    # add each entity
    preIndex = []
    for i in range(len(index)):
        [start, end] = index[i].tolist()
        rawEntity = extendEntity(entities[i],nclass)
        entityIndex = util.findIndex(rawEntity, coeff=coeff)
        # ignore 'other' entity
        if 0 in entityIndex:
            assert(len(entityIndex)==1)
            continue
        # add each nested entity
        for ind in entityIndex:
            # in this case, a "I" entity has already been written before
            if ind%2 == 0 and ((ind in preIndex) or (ind-1 in preIndex)):
                continue
            # merge "B" and "I" entities
            i_ind = int(ind/2)*2 + 2
            new_end = end
            j = i+1
            while j<len(index):
                nextIndex = util.findIndex(extendEntity(entities[j],nclass), coeff=coeff)
                if i_ind in nextIndex:
                    new_end = index[j][1]
                    j += 1
                else:
                    break
            anno_id += 1 
            anno = et.SubElement(passage, 'annotation', {'id':str(anno_id)})
            infon = et.SubElement(anno, 'infon', {'key':"type"})
            infon.text = RAWENTITY[int((ind-1)/2)]
            loc = et.SubElement(anno, 'location',{'length':str(new_end-start), 'offset':str(start)})
            txt = et.SubElement(anno, 'text')
            txt.text = corpus[start:new_end]
        preIndex = entityIndex[:]
    tree = et.ElementTree(root)
    tree.write(os.path.join(path,fileName+'.bioc.xml'),encoding='utf-8',xml_declaration=True,short_empty_elements=False)

# restore the original length entity by padding 0
def extendEntity(entity, nclass):
    if len(entity) == 19:
        return entity
    result = [0 for i in range(19)]
    result[0] = entity[0]
    result[nclass] = entity[1]
    result[nclass+1] = entity[2]
    return result
    
def test(nclass=0,P='__data__/MADE-1.0-test/'):
    global anno_id
    anno_id = 0
    corpPath = P + 'corpus'
    indexPath = P + 'index'
    entityPath = P + 'process2_stepThree_entity'
    outpath = P + 'xml'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    flist = os.listdir(corpPath)
    y = gen.genResultBin(P+'process2_stepThree_entity', flist)
    if nclass != 0:
        truth = gen.twoClass(y, [nclass,nclass+1])
    else:
        truth = y
    #print('start checking')
    for i in range(len(truth)):
        #print(flist[i])
        indices = np.load(os.path.join(indexPath, flist[i]+'.npy'))
        assert(len(truth[i])==len(indices))
        with open(os.path.join(corpPath, flist[i])) as src:
            text = src.read().rstrip()
            assert(indices[-1][1]<=len(text))
        writeXML(flist[i], text, indices, truth[i], path=outpath,nclass=nclass)
    
def mainProcess(modelFile, vocabPath,P='__data__/MADE-1.0-test/',nclass=0):
    global anno_id
    anno_id = 0
    corpCutPath = P+'corp_sent_cut100'
    corpPath = P + 'corpus'
    trainPath = P + 'process2_stepFour_corp'
    truthPath = P + 'process2_stepThree_entity'
    indexPath = P + 'index'
    outpath = P + 'pred-xml'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
        
    flist = os.listdir(corpPath)
    with open(vocabPath, 'rb') as handle:   # load vocabulary from file
        vocab = pickle.load(handle)
    # generate x and y
    x = gen.getIndex(trainPath, flist, vocab)
    y = gen.genResultBin(truthPath, flist)
    
    testx, testy =  gen.toSent(flist, trainPath, x, y)
    testData, testResult = gen.padSent(testx, testy, 100)
    # predict
    model = km.load_model(modelFile)
    predict = model.predict(np.array(testData))
    #write xml
    start = 0
    for i in range(len(flist)):
        indices = np.load(os.path.join(indexPath, flist[i]+'.npy'))
        # raw corpus
        with open(os.path.join(corpPath, flist[i])) as src:
            text = src.read().rstrip()
            assert(indices[-1][1]<=len(text))
        # cut corpus
        with open(os.path.join(corpCutPath, flist[i])) as src:
            textCut = src.read().strip().split('\n')
        # extract entities in prediction
        entities = getEntities(textCut, predict, start)
        assert(len(entities)==len(indices))
        writeXML(flist[i], text, indices, entities, path=outpath, nclass=nclass)
        start += len(textCut)
def getEntities(text, predict, start):
    result = []
    for i in range(len(text)):
        words = text[i].strip().split(' ')
        for j in range(len(words)): 
            result.append(predict[start+i][j])
            if words[j]=='.':
                break
    return result
    
if __name__ == '__main__':
    #vocabPath = 'vocab_made_8000.pkl'
    #vocabPath = 'vocab_glove_400000.pkl'
    vocabPath = 'vocab_nodiscard_8000.pkl'
    modelFile = 'model_made_25-epoch.h5'
    #modelFile = 'model_made_90-0_36-epoch-glove-sent.h5'
    #test(nclass=0,P='__data__/MADE2-1.0-test/')
    mainProcess(modelFile, vocabPath, P='__data__/MADE2-1.0-test/', nclass=5)
