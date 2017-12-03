'''
Entities represented by symbols:
    ADE:A Indication:I SSLIF:S Severity:V Drug:D Dose:O Route:R Frequency:F Duration:U
'''
import os
import xml.etree.ElementTree as et

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

def cmp_file(x):
	c = x.split('_')
	return int(c[0]) * 100000 + int(c[1])

def sortedFileList(path):
	flist = os.listdir(path)
	return sorted(flist, key=cmp_file)

# extract entities info from a single xml file, write the entities into a new file
def parseFile(corp, annot, target, xpath):
    # parse xml
    tree = et.parse(annot)
    annotList = tree.getroot().findall(xpath)

    # initialise target text, all characters except ' ' and '\n' are set to 'o' at beginning
    with open(corp, 'rt') as src:
        srcText = src.read()
    tarChar = []
    for c in srcText:
        if c != ' ' and c != '\n' and c != '\t':
            tarChar.append(OTHER)
        else:
            tarChar.append(c)

    #print(len(srcText),len(tarChar))
    # set corresponding entities
    for a in annotList:
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
            if tarChar[offset+i] != ' ':
                tarChar[offset+i] = symbol
            else:
                symbol = symbol.lower()

    # write characters into new file
    tarText = ''
    for c in tarChar:
        tarText += c
    with open(target, 'wt') as tar:
        tar.write(tarText)
	
def toTokenEntities(corpPath, annotPath, xpath = './document/passage/annotation', newPath='__data__/MADE-1.0/entities'):
    if not os.path.exists(newPath):
        os.makedirs(newPath)
    
    flist = os.listdir(corpPath)
    for f in flist:
        parseFile(os.path.join(corpPath, f), os.path.join(annotPath, f+'.bioc.xml'), os.path.join(newPath, f), xpath)

def main():
    toTokenEntities('__data__/MADE-1.0/corpus', '__data__/MADE-1.0/annotations')

if __name__ == '__main__':
    main()
