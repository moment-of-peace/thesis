import os
import numpy as np
'''
from random import shuffle

folder = '2009truth'
flist = os.listdir(folder)
shuffle(flist)
print(flist)
'''
#import preprocessing_made as pre
import my_utils as util
'''
c,e,t = pre.stepThree('   123  1 2  a bb    ', '   aaa  b b  1 22    ')
print(c)
print(t)
'''
def counter(content):
    c = content.split(' ')
    length = len(c)
    for i in range(length):
        if c[i].isdigit():
            if i > 1 and i < length - 2 and len(c[i])>4:
                print(c[i], c[i-2], c[i-1], c[i+1], c[i+2])
            else:
                print(c[i])
            
'''
#util.process_files('__data__/MADE-1.0/process_stepThree_corp', counter)
with open('joint_file','rt') as f:
    c = f.read().split(' ')
s = set()
for e in c:
    s.add(e)
print(len(s))
'''

'''
corpPath = '__data__/MADE-1.0/corpus'
entityPath = '__data__/MADE-1.0/entities'
flist = os.listdir(corpPath)
for f in flist:
    #print(f)
    with open(os.path.join(corpPath, f), 'rt') as src:
        corp = src.read()
    with open(os.path.join(entityPath, f), 'rt') as src:
        entity = src.read()
    corp = ' ' + corp + ' '
    entity = ' ' + entity + ' '
    i = 1
    while i < len(corp)-1:
        if corp[i]=='\n' and entity[i].lower() != 'x' and entity[i-1].lower() != entity[i+1].lower():# and (entity[i-1].lower() != 'x' or entity[i+1].lower() != 'x'):
            print(f, i, entity[i-1], entity[i], entity[i+1])
        i += 1
'''
'''
path = 'results/made-uppercase'
for i in range(10):
    f = 'result_90_%d_10.txt'%(i)
    print(f)
    with open(os.path.join(path, f), 'rt') as src:
        content = src.read()
    with open('joint.txt', 'at') as tar:
        tar.write(content)
'''
# count how many times each entity appears, and the number of each nested entity
def count_entities(path):
    result = [0 for i in range(19)]
    multi_enti = [0 for i in range(19)]
    flist = os.listdir(path)
    for f in flist:
        n = np.load(os.path.join(path, f))
        for num in n:
            index = util.findIndex(util.numToBin(num), thres=1)
            if len(index) == 1:
                result[index[0]] += 1
            else:
                for i in range(len(index)):
                    result[index[i]] += 1
                    multi_enti[index[i]] += 1
    return result, multi_enti