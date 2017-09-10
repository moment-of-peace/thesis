'''
Build word2vec model using gensim
Author: Yi Liu
'''
import sys
import gensim
#import logging

'''
build list of sentences from a corpusFile.
in the corpus, each line stands for a sentence
in this list, each sentence is a list of tokens
'''
def buildSentences(corpusFile):
    sentences = []
    with open(corpusFile, 'r') as corpus:
        for line in corpus:
            sentences.append(line.strip('\n').split(' '))
    return sentences

def main():
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = buildSentences(sys.argv[1])
    # train model
    model = gensim.models.Word2Vec(sentences, size = 60, window = 3, min_count = 4)
    # save into a text file
    model.wv.save_word2vec_format('word2vec_gensim.txt', binary=False)
