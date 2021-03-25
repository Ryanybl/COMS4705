import nltk

from gensim.models import Word2Vec


def train_w2v(window, size, negative_samples):
    words = read_file()
    w2v = Word2Vec(words, window=window, size=size, negative=negative_samples)
    print(len(w2v.wv.vocab))
    w2v.wv.save("word2vec.wordvectors")

def train_w2v_for_bert():
    """
        I use w2v to generate vocab for BERT for the BATS evaluation
    :return:
    """
    words = read_file()
    w2v = Word2Vec(words, window=2, size=768, min_count=3)
    print(len(w2v.wv.vocab))
    w2v.wv.save("word2vec_for_bert.wordvectors")

def read_file():
    file1 = open('C:\\Users\Robert\Documents\Columbia\\4705\hw3\hw3\data\\brown.txt', 'r')
    lines = file1.readlines()
    all_toks = []
    for line in lines:
        all_toks.append(nltk.tokenize.word_tokenize(line.lower()))

    return all_toks

def run_w2v():
    # train_w2v(2, 50, 1)
    # train_w2v(2, 50, 5)
    # train_w2v(2, 50, 15)
    # train_w2v(2, 100, 1)
    # train_w2v(2, 100, 5)
    # train_w2v(2, 100, 15)
    # train_w2v(2, 300, 1)
    # train_w2v(2, 300, 5)
    # train_w2v(2, 300, 15)
    #
    # train_w2v(5, 50, 1)
    # train_w2v(5, 50, 5)
    # train_w2v(5, 50, 15)
    # train_w2v(5, 100, 1)
    # train_w2v(5, 100, 5)
    # train_w2v(5, 100, 15)
    # train_w2v(5, 300, 1)
    # train_w2v(5, 300, 5)
    # train_w2v(5, 300, 15)
    #
    # train_w2v(10, 50, 1)
    # train_w2v(10, 50, 5)
    # train_w2v(10, 50, 15)
    # train_w2v(10, 100, 1)
    # train_w2v(10, 100, 5)
    # train_w2v(10, 100, 15)
    # train_w2v(10, 300, 1)
    # train_w2v(10, 300, 5)
     train_w2v(10, 300, 15)


run_w2v()
train_w2v_for_bert()
