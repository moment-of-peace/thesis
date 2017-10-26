import os
from random import shuffle

folder = '2009truth'
flist = os.listdir(folder)
shuffle(flist)
print(flist)
