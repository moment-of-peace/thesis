import os

def write_list(l, fileName=None, linefeed='\n'):
    string = ''
    for e in l:
        string = string + str(e) + linefeed
    if fileName != None:
        with open(fileName,'wt') as tar:
            tar.write(string)
    return string
            
def write_dict(d, fileName=None, linefeed='\n'):
    string = ''
    for k in d.keys():
        string = '%s%s,%s%s'%(string, str(k), str(d[k]), linefeed)
    if fileName != None:
        with open(fileName,'wt') as tar:
            tar.write(string)
    return string
    

# write a file into a directory, create the path if not exists
def write_path_file(path, fileName, content, flag='wt'):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, fileName), flag) as tar:
        tar.write(content)
        
# do some processing for each file in a directory
def process_files(path, func):
    flist = os.listdir(path)
    for f in flist:
        with open(os.path.join(path, f), 'rt') as src:
            content = src.read()
        func(content)
        
# join all files in a directory and form a single file
def join_files(path, fileName='joint_file.txt', separator=''):
    flist = os.listdir(path)
    sep = ''
    for f in flist:
        print(f)
        with open(os.path.join(path, f), 'rt') as src:
            content = src.read()
        with open(fileName, 'at') as tar:
            tar.write(sep + content)
        sep = separator
        
if __name__ == '__main__':
    join_files('__data__/MADE-1.0/process_stepFour_corp', separator=' . ')