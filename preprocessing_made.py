'''
Entities represented by symbols:
    ADE:A Indication:I SSLIF:S Severity:V Drug:D Dose:O Route:R Frequency:F Duration:U
'''
import os
import xml.etree.ElementTree as et

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
OTHER = 'X'

TAG_ENTITY = 'infon'
TAG_LOC = 'location'
TAG_TEXT = 'text'
ATTR_OFF = 'offset'
ATTR_LEN = 'length'

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
            symbol = OTHER
            print(corp, entity, a.tag)
        # position
        loc = a.find(TAG_LOC)
        offset = int(loc.get(ATTR_OFF))
        length = int(loc.get(ATTR_LEN))
        # set entity infomation
        for i in range(length):
            if srcText[offset+i].isspace():
                symbol = symbol.lower()
            tarChar[offset+i] = symbol
            '''
            if tarChar[offset+i] != ' ':
                tarChar[offset+i] = symbol
            else:
                symbol = symbol.lower()
            '''
    # write characters into new file
    tarText = ''
    for c in tarChar:
        tarText += c
    with open(target, 'wt') as tar:
        tar.write(tarText)
	
def toTokenEntities(corpPath, annoPath, xpath = './document/passage/annotation', newPath='__data__/MADE-1.0/entities'):
    if not os.path.exists(newPath):
        os.makedirs(newPath)
    
    flist = os.listdir(corpPath)
    for f in flist:
        parseFile(os.path.join(corpPath, f), os.path.join(annoPath, f+'.bioc.xml'), os.path.join(newPath, f), xpath)

# preprocessing for corpus and annotations
def preprocesses(corpPath, entityPath, steps, newPath='__data__/MADE-1.0/process'):
    flist = os.listdir(corpPath)
    for f in flist:
        print(f)
        with open(os.path.join(corpPath, f), 'rt') as src:
            corp = src.read()
        with open(os.path.join(entityPath, f), 'rt') as src:
            entity = src.read()
        # check which steps to implement
        if 1 in steps:
            corp, entity = processStep(corp, entity, stepOne, f, newPath)
        if 2 in steps:
            corp, entity = processStep(corp, entity, stepTwo, f, newPath)
        if 3 in steps:
            corp, entity = processStep(corp, entity, stepThree, f, newPath)
        if 4 in steps:
            corp = stepFour(corp, f, newPath)

# delete some comas and dots, to lower case, replace \n \t with spaces
def processStep(corp, entity, stepFunc, fileName, newPath='__data__/MADE-1.0/process'):
    # avoid index out of range
    corp = ' ' + corp + ' '
    entity = ' ' + entity + ' '
    c, e, trace = stepFunc(corp, entity)
    newCorp = ''.join(c)
    newEntity = ''.join(e)
    # write processed corpus, entity, and processing trace to new files
    util.write_path_file('%s_%s_corp'%(newPath,stepFunc.__name__), fileName, newCorp)
    util.write_path_file('%s_%s_entity'%(newPath,stepFunc.__name__), fileName, newEntity)
    util.write_path_file('%s_%s_trace'%(newPath,stepFunc.__name__), fileName, util.write_list(trace))
    
    return newCorp, newEntity

# the first step of preprocessing: remove "," and "." in numbers, put spaces at positions of '\t', '\n', and ' '
def stepOne(corp, entity):
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
            e.append(' ')
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
def stepTwo(corp, entity):
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
                e.append(' ')
                trace.append((i-1, 1))
        elif pre != 0 and (corp[i] == '.' or corp[i] == ',' or corp[i] == ';' or corp[i] == ')'):
            c.append(' ')
            e.append(' ')
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
def stepThree(corp, entity):
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
                e.append(' ')
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
def checkCorpEnti(corpPath, entityPath, flag):
    flist = os.listdir(corpPath)
    flist2 = os.listdir(entityPath)
    assert(len(flist) == 876)
    assert(len(flist2) == 876)
    
    for f in flist:
        #print(f)
        with open(os.path.join(corpPath, f), 'rt') as src:
            corp = src.read()
        with open(os.path.join(entityPath, f), 'rt') as src:
            entity = src.read()
        assert(len(corp) == len(entity))
        
        if flag:
            for i in range(len(corp)):
                if (corp[i]==' ' and entity[i]!=' ') or (corp[i]!=' ' and entity[i]==' '):
                    print('  wrong space at', i)
                    exit()
    print('checks finished')

def main():
    P = '__data__/MADE-1.0/'
    
    toTokenEntities(P+'corpus', P+'annotations')
    preprocesses(P+'corpus', P+'entities', [1,2,3,4])
    
    checkCorpEnti(P+'corpus', P+'entities', False)
    checkCorpEnti(P+'process_stepOne_corp', P+'process_stepOne_entity', True)
    checkCorpEnti(P+'process_stepTwo_corp', P+'process_stepTwo_entity', True)
    checkCorpEnti(P+'process_stepThree_corp', P+'process_stepThree_entity', True)
    checkCorpEnti(P+'process_stepFour_corp', P+'process_stepThree_entity', True)
    
    
if __name__ == '__main__':
    main()
