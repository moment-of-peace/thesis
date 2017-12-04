import os

def cmp_file(x):
	c = x.split('_')
	return int(c[0]) * 100000 + int(c[1])

def sortedFileList(path):
	flist = os.listdir(path)
	return sorted(flist, key=cmp_file)
	
def toTokenEntities(path, newPath='__data__/MADE-1.0/token_entities'):
	if not os.path.exists(newpath):
        os.makedirs(newpath)
	