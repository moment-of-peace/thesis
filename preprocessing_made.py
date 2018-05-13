'''
Entities represented by symbols:
    ADE:A Indication:I SSLIF:S Severity:V Drug:D Dose:O Route:R Frequency:F Duration:U
'''
import os
import sys
import xml.etree.ElementTree as et
import numpy as np

import my_utils as util

ENTITY_DIC = {'ADE':'A',
              'Indication':'I',
              'SSLIF':'S',
              'Severity':'V',
              'Drug':'D', 
              'Dose':'O',
              'Route':'R',
              'Frequency':'F',
              'Duration':'U'}
ENTITY_INDEX={'ADE':1,
              'Indication':3,
              'SSLIF':5,
              'Severity':7,
              'Drug':9, 
              'Dose':11,
              'Route':13,
              'Frequency':15,
              'Duration':17}
OTHER = 'X'

TAG_ENTITY = 'infon'
TAG_LOC = 'location'
TAG_TEXT = 'text'
ATTR_OFF = 'offset'
ATTR_LEN = 'length'
indexPath = '__data__/index'

def parseFileBin(corp, anno, target, xpath):
    # parse xml
    tree = et.parse(anno)
    annoList = tree.getroot().findall(xpath)

    # initialise target text, all characters except ' ' and '\n' are set to OTHER at beginning
    with open(corp, 'rt') as src:
        srcText = src.read()
    n = np.ones(len(srcText), dtype=np.uint32)

    #print(len(srcText),len(tarChar))
    # set corresponding entities
    for a in annoList:
        # entity and its symbol
        entity = a.find(TAG_ENTITY).text
        try:
            index = ENTITY_INDEX[entity]
        except:
            print(corp, entity, a.tag)
            continue
        # position
        bits = index
        loc = a.find(TAG_LOC)
        offset = int(loc.get(ATTR_OFF))
        length = int(loc.get(ATTR_LEN))
        # set entity infomation
        for i in range(offset, offset+length):
            if srcText[i].isspace():
                bits = index + 1
            if n[i] == 1:
                n[i] = 0
            n[i] |= (1<<bits)
            
    # write characters into new file
    np.save(target, n)

# extract entities info from a single xml file, write the entities into a new file
def parseFile(corp, anno, target, xpath):
    # parse xml
    tree = et.parse(anno)
    annoList = tree.getroot().findall(xpath)

    # initialise target text, all characters except ' ' and '\n' are set to OTHER at beginning
    with open(corp, 'rt') as src:
        srcText = src.read()
    tarChar = []
    for c in srcText:
        tarChar.append(OTHER)
        '''
        if c != ' ' and c != '\n' and c != '\t':
            tarChar.append(OTHER)
        else:
            tarChar.append(c)
        '''

    #print(len(srcText),len(tarChar))
    # set corresponding entities
    for a in annoList:
        # entity and its symbol
        entity = a.find(TAG_ENTITY).text
        try:
            symbol = ENTITY_DIC[entity]
        except:
            #symbol = OTHER
            print(corp, entity, a.tag)
            continue
        # position
        loc = a.find(TAG_LOC)
        offset = int(loc.get(ATTR_OFF))
        length = int(loc.get(ATTR_LEN))
        # set entity infomation
        for i in range(length):
            if srcText[offset+i].isspace():
                symbol = symbol.lower()
            tarChar[offset+i] = symbol
            
    # write characters into new file
    with open(target, 'wt') as tar:
        tar.write(''.join(tarChar))
    
def toTokenEntities(corpPath, annoPath, xpath = './document/passage/annotation', newPath='__data__/MADE-1.0/entities'):
    if not os.path.exists(newPath):
        os.makedirs(newPath)
    
    flist = os.listdir(corpPath)
    for f in flist:
        parseFile(os.path.join(corpPath, f), os.path.join(annoPath, f+'.bioc.xml'), os.path.join(newPath, f), xpath)

def toBinEntities(corpPath, annoPath, xpath = './document/passage/annotation', newPath='__data__/MADE-1.0/entities2'):
    if not os.path.exists(newPath):
        os.makedirs(newPath)
    
    flist = os.listdir(corpPath)
    for f in flist:
        parseFileBin(os.path.join(corpPath, f), os.path.join(annoPath, f+'.bioc.xml'), os.path.join(newPath, f), xpath)

# preprocessing for corpus and annotations
def preprocesses(corpPath, entityPath, steps, spaceChar, newPath='__data__/MADE-1.0/process'):
    flist = os.listdir(corpPath)
    for f in flist:
        print(f)
        with open(os.path.join(corpPath, f), 'rt') as src:
            corp = src.read()
        if spaceChar == 0:
            entity = np.load(os.path.join(entityPath, f+'.npy'))
        else:
            with open(os.path.join(entityPath, f), 'rt') as src:
                entity = src.read()
        # check which steps to implement
        if 1 in steps:
            corp, entity = processStep(corp, entity, stepOne, f, spaceChar, newPath)
        if 2 in steps:
            corp, entity = processStep(corp, entity, stepTwo, f, spaceChar, newPath)
        if 3 in steps:
            corp, entity = processStep(corp, entity, stepThree, f, spaceChar, newPath)
        if 4 in steps:
            corp = stepFour(corp, f, newPath)

# delete some comas and dots, to lower case, replace \n \t with spaces
def processStep(corp, entityTemp, stepFunc, fileName, spaceChar, newPath='__data__/MADE-1.0/process'):
    path = os.path.join('%s_%s_entity'%(newPath,stepFunc.__name__))
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(indexPath):
        os.makedirs(indexPath)
    # avoid index out of range
    corp = ' ' + corp + ' '
    if spaceChar == 0:
        entity = np.zeros(len(entityTemp) + 2, dtype=np.uint32)
        entity[1:-1] = entityTemp[:]
    else:
        entity = ' ' + entity + ' '

    c, e, trace = stepFunc(fileName, corp, entity, spaceChar)
    newCorp = ''.join(c)

    # write processed corpus, entity, and processing trace to new files
    util.write_path_file('%s_%s_corp'%(newPath,stepFunc.__name__), fileName, newCorp)
    util.write_path_file('%s_%s_trace'%(newPath,stepFunc.__name__), fileName, util.write_list(trace))
    if spaceChar == 0:
        newEntity = np.array(e, dtype=np.uint32)
        np.save(os.path.join('%s_%s_entity'%(newPath,stepFunc.__name__), fileName), newEntity)
    else:
        newEntity = ''.join(e)
        util.write_path_file('%s_%s_entity'%(newPath,stepFunc.__name__), fileName, newEntity)
    
    return newCorp, newEntity

# the first step of preprocessing: remove "," and "." in numbers, put spaces at positions of '\t', '\n', and ' '
def stepTwo(fileName, corp, entity, spaceChar):
    c, e, trace = [], [], []
    i = 1
    while i < len(corp)-1:
        if corp[i]==',' and corp[i-1].isdigit() and corp[i+1].isdigit():
            trace.append((i-1, -1))
        elif corp[i]=='.' and corp[i-1].isdigit() and corp[i+1].isdigit():
            num = countDigits(corp, i+1)
            trace.append((i-1, -num))
            i += num
            continue
        elif corp[i].isspace():
            c.append(' ')
            e.append(spaceChar)
        else:
            c.append(corp[i].lower())
            e.append(entity[i])
        i += 1
    return c, e, trace

# count how many digits after a dots
def countDigits(corp, i):
    num = 1
    while corp[i].isdigit():
        num += 1
        i += 1
    return num

# insert spaces to split letters, numbers, and other signals
def stepOne(fileName, corp, entity, spaceChar):
    c, e, trace, index = [], [], [], []
    stops = ['.',',',';',':','(',')','[',']','?','!']
    i = 1
    token_start = 0
    pre = 0 # 0: space, 1: letter, 2: number, 3: other
    # c = ' qwe,df.f.ff, ff.ff.ff, f.f.p.a&kjl q.w.e.33.3 45,9 '
    length = len(corp)
    while i < length-1:
        spaceFlag = False   # indicate whether an extra space is inserted
        cur = represent(corp[i])
        if pre != 0 and i < length-11 and corp[i:i+10].lower() == 'additional':
            appendSpace(c,e,trace,index,spaceChar,token_start,i)
            spaceFlag = True
            token_start = i-1
        elif cur == 1:
            if corp[i-1] == '.' and corp[i-2].isalpha() and corp[i+1] == '.':
                pass
            elif pre == 2 or pre == 3:
                appendSpace(c,e,trace,index,spaceChar,token_start,i)
                spaceFlag = True
                token_start = i-1
        elif cur == 2:
            if corp[i-1] in {',','.'} and corp[i-2].isdigit():
                pass
            elif pre == 1 or pre == 3:
                appendSpace(c,e,trace,index,spaceChar,token_start,i)
                spaceFlag = True
                token_start = i-1
        elif corp[i] == ',' and corp[i-1].isdigit() and corp[i+1].isdigit():
            pass
        elif corp[i] == '.':
            if corp[i-1].isalpha() and corp[i-2] == '.':
                pass
            elif corp[i-1].isalpha() and corp[i+1].isalpha() and corp[i+2] == '.':
                pass
            elif corp[i-1].isdigit() and corp[i+1].isdigit():
                pass
            else:
                appendSpace(c,e,trace,index,spaceChar,token_start,i)
                spaceFlag = True
                token_start = i-1
        elif cur == 3:
            if (pre not in {0,3}) or (corp[i] in {',','.'} and pre != 0):
                appendSpace(c,e,trace,index,spaceChar,token_start,i)
                spaceFlag = True
                token_start = i-1
        elif cur == 0:
            if token_start != (i-1):
                index.append([token_start,i-1])
            entity[i] = 0
            token_start = i
        '''
        if cur != 0 and corp[i-1] in stops:
            c.append(' ')
            e.append(spaceChar)
            trace.append((i-1, 1))
        elif pre != 0 and cur != 0 and pre != cur:
            # words such as "a.m." should not be split
            if corp[i] == '.' and corp[i-1].isalpha() and corp[i+1].isalpha():
                if (i > 1 and corp[i-2].isalpha()) or (i < len(corp)-2 and corp[i+2].isalpha()):
                    c.append(' ')
                    e.append(spaceChar)
                    trace.append((i-1, 1))
            elif corp[i] == '.' and corp[i-1].isalpha() and i > 1 and corp[i-2] == '.':
                pass
            elif cur == 1 and corp[i-1] == '.' and i > 1 and corp[i-2].isalpha():
                pass
            else:
                c.append(' ')
                e.append(spaceChar)
                trace.append((i-1, 1))
        elif pre != 0 and (corp[i] in stops):
            c.append(' ')
            e.append(spaceChar)
            trace.append((i-1, 1))
        '''
        '''
        if pre != 0:
            if (cur != 0 and pre != cur) or cur == 3:
                c.append(' ')
                e.append(' ')
                trace.append((i-1, 1))
        '''
        c.append(corp[i])
        if spaceFlag:
            e.append(BtoIentity(entity[i]))
        else:
            e.append(entity[i])
        pre = cur
        i += 1
    # the start and end index of the final token
    if token_start != (len(corp)-2):
        index.append([token_start,len(corp)-2])
    np.save(os.path.join(indexPath,fileName), np.array(index))
    return c, e, trace
    
# use 0, 1, 2, 3, to represent a char. 0: space, 1: letter, 2: number, 3: other
def represent(chara):
    if chara.isspace():
        return 0
    if chara.isalpha():
        return 1
    if chara.isdigit():
        return 2
    return 3
# append space to corp, entity, trace log
def appendSpace(c,e,trace,index,spaceChar,token_start,i):
    c.append(' ')
    e.append(spaceChar)
    trace.append((i-1, 1))
    if token_start != (i-1):
        index.append([token_start,i-1])
# after insert a space, convert the "B" entity to "I"
def BtoIentity(num):
    if num in {0,1}:
        return num
    array = util.numToBin(num)
    index = util.findIndex(array, thres=1)
    result = []
    for n in index:
        if n%2 == 1:
            if n+1 not in index:
                result.append(n+1)
        else:
            result.append(n)
    newNum = 0
    for n in result:
        newNum += 1<<n
    return newNum
    
# remove duplicated spaces (including start and tail spaces)
def stepThree(fileName, corp, entity, spaceChar):
    c, e, trace = [], [], []
    i = 1
    flag = (0, True)
    num = 0 # how many spaces to remove
    while i < len(corp)-1:
        if corp[i].isspace():
            if flag[1]:
                num -= 1
            else:
                flag = (i, True)
                num = 0
                c.append(' ')
                e.append(spaceChar)
        else:
            if flag[1]:
                if num < 0:
                    trace.append((flag[0], num))
                flag = (-1, False)
            c.append(corp[i])
            e.append(entity[i])   
        i += 1
    # handle tail space
    if flag[1]:
        c.pop()
        e.pop()
        trace.append((flag[0]-1, num-1))
    return c, e, trace

# convert numbers
def stepFour(corp, fileName, newPath):
    words = corp.split(' ')
    newCorp = ''
    for w in words:
        if w.isdigit():
            newCorp += convertNum(w) + ' '
        else:
            newCorp += w + ' '
    newCorp = newCorp[:-1]  # remove tail space
    util.write_path_file(newPath+'_stepFour_corp', fileName, newCorp)
    return newCorp

# convert a number
def convertNum(word):
    result = int(int(word[0])/5)*5 + 1
    result = str(result)
    for i in range(len(word)-1):
        result += '0'
    return result
    
# check corpus and entities
def checkCorpEnti(corpPath, entityPath, flag, spaceChar, fileNum):
    flist = os.listdir(corpPath)
    flist2 = os.listdir(entityPath)
    # check number of files
    assert(len(flist) == fileNum)
    assert(len(flist2) == fileNum)
    
    for f in flist:
        #print(f)
        with open(os.path.join(corpPath, f), 'rt') as src:
            corp = src.read()
        if spaceChar == 0:
            entity = np.load(os.path.join(entityPath, f+'.npy'))
        else:
            with open(os.path.join(entityPath, f), 'rt') as src:
                entity = src.read()
        # check file length
        assert(len(corp) == len(entity))
        
        if flag:
            for i in range(len(corp)):
                # check spaces
                if (corp[i].isspace() and entity[i]!=spaceChar) or ((not corp[i].isspace()) and entity[i]==spaceChar):
                    print('  wrong space at', i)
                    exit()
    print('checks finished')

# sentences level tokenize
def toSentence(path, newPath):
    if not os.path.exists(newPath):
        os.makedirs(newPath)
        
    flist = os.listdir(path)
    for f in flist:
        newText = ''
        with open(os.path.join(path, f)) as src:
            text = src.read()
            start = 0
            index = text.find(' . ')
            while index != -1:
                newText += (text[start:index] + ' .\n')
                start = index + 3
                index = text.find(' . ', index+1)
            newText += text[start:]
            with open(os.path.join(newPath, f), 'wt') as tar:
                tar.write(newText)

# the max length of sentences
def max_length(path):
    max_len = 100
    name = ''
    index = 0
    text = ''
    
    flist = os.listdir(path)
    for f in flist:
        with open(os.path.join(path,f)) as src:
            sentences = src.read().strip().split('\n')
            for i in range(len(sentences)):
                length = len(sentences[i].split(' '))
                if length > max_len:
                    text += (sentences[i] + '\n')
                    max_len = length
                    #index = i
                    #name = f
    print(max_len, name, i)
    with open('long_sent.txt','wt') as tar:
        tar.write(text)

# cut sentences in corp
def cutSentCorp(path, length, newPath):
    if not os.path.exists(newPath):
        os.makedirs(newPath)
    flist = os.listdir(path)
    for f in flist:
        textCut = ''
        with open(os.path.join(path,f)) as src:
            text = src.read().strip().split('\n')
            for s in text:
                sent = s.strip().split(' ')
                if len(sent) <= length:
                    textCut += (' '.join(sent)+'\n')
                else:
                    for i in range(int(len(sent)/length)):
                        start = i * length
                        textCut += (' '.join(sent[start:start+length])+'\n')
                    m = len(sent)%length
                    if m != 0:
                        textCut += (' '.join(sent[-m:])+'\n')
        with open(os.path.join(newPath,f),'wt') as tar:
            tar.write(textCut)
            
def mainProcess(P, fileNum, spaceChar = 0):    
    toBinEntities(P+'corpus', P+'annotations', newPath=P+'entities')
    preprocesses(P+'corpus', P+'entities', [1,2,3,4], spaceChar, newPath=P+'process2')
    
    checkCorpEnti(P+'corpus', P+'entities', False, spaceChar, fileNum)
    checkCorpEnti(P+'process2_stepOne_corp', P+'process2_stepOne_entity', True, spaceChar, fileNum)
    checkCorpEnti(P+'process2_stepTwo_corp', P+'process2_stepTwo_entity', True, spaceChar, fileNum)
    checkCorpEnti(P+'process2_stepThree_corp', P+'process2_stepThree_entity', True, spaceChar, fileNum)
    checkCorpEnti(P+'process2_stepFour_corp', P+'process2_stepThree_entity', True, spaceChar, fileNum)
    
    
if __name__ == '__main__':
    if sys.argv[1] == '1':
        P = '__data__/MADE2-1.0/'
        fileNum = 876   #876 213
    else:
        P = '__data__/MADE2-1.0-test/'
        fileNum = 213   #876 213
    mainProcess(P, fileNum)
    toSentence(P + 'process2_stepFour_corp',P + 'corp_sentence')
    cutSentCorp(P + 'corp_sentence',100,P + 'corp_sent_cut%d'%100)
    #max_length('__data__/MADE-1.0-test/corp_sent_cut100')
