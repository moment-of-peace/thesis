import sys
import getopt
from sklearn.metrics import f1_score

def f1_eval(fileName, labels):
    tru = []
    pre = []
    with open(fileName,'rt') as src:
        for line in src:
            content = line.strip('\n').split(' ')
            #if content[0] != '19' or content[1] != '19':
            tru.append(int(content[0]))
            pre.append(int(content[1]))
    print('f1 score for each label:\n', f1_score(tru, pre, labels=labels, average=None))
    print('\nmicro f1 score:', f1_score(tru, pre, labels=labels, average='micro'))
    print('macro f1 score:', f1_score(tru, pre, labels=labels, average='macro'))

def main():
    fileName = '__out__/result_timedis_bi_20e.txt'
    labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]

    options,args = getopt.getopt(sys.argv[1:],"f:d:b:l:m:e:")
    for opt, para in options:
        if opt == '-f':
            fileName = para
    
    f1_eval(fileName, labels)

if __name__ == '__main__':
    main()
