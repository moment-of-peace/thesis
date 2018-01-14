import os
import my_utils as util

entityDict = {'A':'B-ADE',
              'a':'I-ADE',
              'I':'B-Indication',
              'i':'I-Indication',
              'S':'B-SSLIF',
              's':'I-SSLIF',
              'V':'B-Severity',
              'v':'I-Severity',
              'D':'B-Drugname',
              'd':'I-Drugname',
              'O':'B-Dosage',
              'o':'I-Dosage',
              'R':'B-Route',
              'r':'I-Route',
              'F':'B-Frequency',
              'f':'I-Frequency',
              'U':'B-Duration',
              'u':'I-Duration',
              'X':'O'}
              
def gen_conll_output(truthPath, predPath, corpPath='__data__/MADE-1.0/corpus/', outPath='__data__/MADE-1.0/'):
    truthList = os.listdir(truthPath)
    predList = os.listdir(predPath)
    corpList = os.listdir(corpPath)
    assert(len(truthList) == len(predList))
    assert(len(truthList) == len(corpList))
    
    for f in truthList:
        with open(os.path.join(truthPath, f)) as src:
            truth = src.read()
        with open(os.path.join(predPath, f)) as src:
            pred = src.read()
        with open(os.path.join(corpPath, f)) as src:
            corp = src.read()
            
        util.write_path_file(outPath, f, convert_conll(f, truth, pred, corp))
        
# process truth and prediction of a single corpus file
def convert_conll(f, truth, pred, corp):
    corp = ' ' + corp + '  '
    start = 0
    pre = ' '
    result = ''
    
    for i in range(1, len(corp)-1):
        if (not corp[i].isspace()) and start == 0:
            start = i
        if corp[i].isspace() and (not corp[i-1].isspace()):
            result += '%s valid_text_00000 %d %d %s %s %s\n'%(corp[start:i],start-1,i-1,f,entityDict[truth[start-1]],entityDict[pred[start-1]])
            start = 0
        elif is_stop(corp, i):
            result += '%s valid_text_00000 %d %d %s %s %s\n'%(corp[start:i],start-1,i-1,f,entityDict[truth[start-1]],entityDict[pred[start-1]])
            start = i
    return result
    
# is stop of a token/phrase/sentence
def is_stop(corp, i):
    if corp[i] in ['.',','] and corp[i-1].isdigit() and corp[i+1].isdigit():
            return False
    if corp[i] == '/':
        if not(corp[i-1].isspace() or corp[i+1].isspace()):
            return False
            
    return True
        