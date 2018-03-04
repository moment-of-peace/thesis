'''
Entities represented by symbols:
    ADE:A Indication:I SSLIF:S Severity:V Drug:D Dose:O Route:R Frequency:F Duration:U
'''
import os
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
ENTITY_INDEX={'ADE':0,
              'Indication':2,
              'SSLIF':4,
              'Severity':6,
              'Drug':8, 
              'Dose':10,
              'Route':12,
              'Frequency':14,
              'Duration':16}
OTHER = 'X'

TAG_ENTITY = 'infon'
TAG_LOC = 'location'
TAG_TEXT = 'text'
ATTR_OFF = 'offset'
ATTR_LEN = 'length'

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
            n[i] |= (2<<bits)
            
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
    # avoid index out of range
    corp = ' ' + corp + ' '
    if spaceChar == 0:
        entity = np.zeros(len(entityTemp) + 2)
        entity[1:-1] = entityTemp[:]
    else:
        entity = ' ' + entity + ' '

    c, e, trace = stepFunc(corp, entity, spaceChar)
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
def stepOne(corp, entity, spaceChar):
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
        elif corp[i]=='\t' or corp[i]=='\n' or corp[i]==' ':
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
def stepTwo(corp, entity, spaceChar):
    c, e, trace = [], [], []
    i = 1
    pre = 0 # 0: space, 1: letter, 2: number, 3: other
    while i < len(corp)-1:
        cur = represent(corp[i])
        
        if pre != 0 and cur != 0 and pre != cur:
            # words such as "a.m." should not be split
            if corp[i] == '.' and corp[i-1].isalpha() and corp[i+1].isalpha():
                pass
            elif corp[i] == '.' and corp[i-1].isalpha() and i > 1 and corp[i-2] == '.':
                pass
            elif cur == 1 and corp[i-1] == '.' and i > 1 and corp[i-2].isalpha():
                pass
            else:
                c.append(' ')
                e.append(spaceChar)
                trace.append((i-1, 1))
        elif pre != 0 and (corp[i] == '.' or corp[i] == ',' or corp[i] == ';' or corp[i] == ')'):
            c.append(' ')
            e.append(spaceChar)
            trace.append((i-1, 1))
        '''
        if pre != 0:
            if (cur != 0 and pre != cur) or cur == 3:
                c.append(' ')
                e.append(' ')
                trace.append((i-1, 1))
        '''
        c.append(corp[i])
        e.append(entity[i])
        pre = cur
        i += 1
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

# remove duplicated spaces (including start and tail spaces)
def stepThree(corp, entity, spaceChar):
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
def checkCorpEnti(corpPath, entityPath, flag, spaceChar):
    flist = os.listdir(corpPath)
    flist2 = os.listdir(entityPath)
    # check number of files
    assert(len(flist) == 876)
    assert(len(flist2) == 876)
    
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
                if (corp[i]==' ' and entity[i]!=spaceChar) or (corp[i]!=' ' and entity[i]==spaceChar):
                    print('  wrong space at', i)
                    exit()
    print('checks finished')

def main():
    P = '__data__/MADE-1.0/'
    spaceChar = 0
    
    #toBinEntities(P+'corpus', P+'annotations')
    #preprocesses(P+'corpus', P+'entities2', [1,2,3,4], spaceChar, newPath='__data__/MADE-1.0/process2')
    
    checkCorpEnti(P+'corpus', P+'entities2', False, spaceChar)
    checkCorpEnti(P+'process2_stepOne_corp', P+'process2_stepOne_entity', True, spaceChar)
    checkCorpEnti(P+'process2_stepTwo_corp', P+'process2_stepTwo_entity', True, spaceChar)
    checkCorpEnti(P+'process2_stepThree_corp', P+'process2_stepThree_entity', True, spaceChar)
    checkCorpEnti(P+'process2_stepFour_corp', P+'process2_stepThree_entity', True, spaceChar)
    
    
if __name__ == '__main__':
    main()
