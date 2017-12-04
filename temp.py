'''
import os
from random import shuffle

folder = '2009truth'
flist = os.listdir(folder)
shuffle(flist)
print(flist)
'''
import preprocessing_made as pre
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
            
#util.process_files('__data__/MADE-1.0/process_stepThree_corp', counter)
with open('joint_file','rt') as f:
    c = f.read().split(' ')
s = set()
for e in c:
    s.add(e)
print(len(s))