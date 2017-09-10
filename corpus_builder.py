import os
import sys
import string

def searchFiles(path, fid):
    for doc in os.listdir(path):
        docPath = os.path.join(path, doc)
        if os.path.isdir(docPath):  # if still a dir, search recursively
            searchFiles(docPath, fid)
        else:
            writeCorpus(docPath, fid)   # extract corpus from a file

def writeCorpus(docPath, fid):
    src = open(docPath, 'rt')
    content = src.readline()
    while content != '':
        result = clean(content) # to lower case, remove signals, numbers
        if result != '':
            fid.write(result + '\n')  # form into sentences
            # fid.write(result + ' ')   # in one line
        content = src.readline()
    src.close()

def clean(content):
    result = ''
    content = content.strip('\n').lower()
    for c in content:
        if c in string.ascii_letters:
            result = result + c
        elif c == ' ':
            if result != '' and result[-1] != ' ':
                result = result + c
        '''
        if c not in string.punctuation:
            if c != ' ':
                result = result + c
            elif result != '' and result[-1] != ' ':
                result = result + c
        '''
    
    return result.strip(' ')

def main():
    fid = open('my_corpus.txt', 'wt')
    path = sys.argv[1]
    searchFiles(path, fid)
    fid.close()

if __name__ == '__main__':
    main()
