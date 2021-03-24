# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#!pip install transformers
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np 
import scipy as sp 
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest
from scipy.sparse import lil_matrix
import pandas as pd 


# %%
from evaluate import *
from process import *


# %%
# f = open("data/brown.txt")
# vectorizer = CountVectorizer()
# vectorizer.fit(f)
# vocab = vectorizer.vocabulary_
# tokenizer = vectorizer.build_tokenizer()


# %%
def file_to_words(filename):
    with open (filename, "r") as myfile:
        lines=myfile.readlines()
        return lines
# train_lines = file_to_words("data/brown.txt")


# %%
def cooc_matrix(train_lines, context_window, vocab, tokenizer):
    if len(train_lines) == 0:
        return []
    r = len(vocab.keys())
    D = lil_matrix((r,r))
    for i in range(len(train_lines)):
        line = tokenizer(train_lines[i])
        print(f"Generating cooc_matrix on line {i+1}: {round(i/len(train_lines)*100,1)} percent complete", end="\r")
        for j in range(len(line)):
            word = line[j].lower()
            word_index = vocab[word]
            for k in range(1,context_window+1):
                if j+k < len(line):
                    context_index = vocab[line[j+k].lower()]
                    if not context_index in D.rows[word_index]:
                        D.rows[word_index].append(context_index)
                        D.data[word_index].append(1)
                    else:
                        D.data[word_index][D.rows[word_index].index(context_index)] += 1
                if j-k >= 0:
                    context_index = vocab[line[j-k].lower()]
                    if not context_index in D.rows[word_index]:
                        D.rows[word_index].append(context_index)
                        D.data[word_index].append(1)
                    else:
                        D.data[word_index][D.rows[word_index].index(context_index)] += 1
    return D


# %%
# D = cooc_matrix(train_lines,2,vocab, tokenizer)


# %%
# D_sum = D.sum(1).A
# nD = D.sum()


# %%
def D_to_M(D,D_sum,nD):
    n = len(D_sum)
    print("creating PMI as lil_matrix")
    M = D.copy()
    rows = M.rows
    data = M.data
    for i in range(len(rows)):
        print(f"Generating PMI: {round(i/len(rows)*100,1)} percent complete", end="\r")
        row = rows[i]
        for j in range(len(row)):
            # row[j] = index in the vocabulary dictionary for c, i = index for w
            assert data[i][j] != 0
            data[i][j] = np.log(data[i][j] * nD / D_sum[i][0] / D_sum[row[j]][0])
            assert data[i][j] != 0
    print("PMI created")
    return M
    


# %%
# M = D_to_M(D,D_sum,nD)


# %%
def M_to_svd(M,dimension):
    svd = TruncatedSVD(n_components=dimension)
    U = svd.fit_transform(M)
    V = svd.components_
    S = svd.singular_values_ 
    return U,S,V


# %%
# u,s_diag,vt = M_to_svd(M,50)
# s = np.zeros((len(s_diag),len(s_diag)))
# np.fill_diagonal(s,s_diag)
# print(u.shape,s.shape,vt.shape)


# %%
# W = np.matmul(u,np.sqrt(s))
# print(W.shape)


# %%



# %%
def write_results(vocab,W,filename):
    embedding = {}
    for word in vocab.keys():
        embedding[word] = W[vocab[word]]
    f = open(filename,"w", encoding='utf8')
    for key, value in embedding.items():
        if not np.any(value):
            continue
        values = ""
        for i in range(len(value)):
            values += f"{value[i]} "
        values = values.strip()
        f.write("{0} {1}\n".format(str(key), values))
    print(f.name)
    f.close()


# %%
# tokenizer("S. J. Perelman")


# %%
train_path = "data/brown.txt"
def generate_all_embeddings(train_path):
    f = open(train_path)
    vectorizer = CountVectorizer()
    vectorizer.fit(f)
    vocab = vectorizer.vocabulary_
    tokenizer = vectorizer.build_tokenizer()
    train_lines = file_to_words(train_path)
    for context_window in [2,5,10]:
        D = cooc_matrix(train_lines,context_window,vocab, tokenizer)
        D_sum = D.sum(1).A
        nD = D.sum()
        M = D_to_M(D,D_sum,nD)

        for dimension in [50,100,300]:
            u,s_diag,vt = M_to_svd(M,dimension)
            s = np.zeros((len(s_diag),len(s_diag)))
            np.fill_diagonal(s,s_diag)
            W = np.matmul(u,np.sqrt(s))
            filename = f"svd_{dimension}_{context_window}.txt"
            write_results(vocab,W,filename)


# %%
generate_all_embeddings(train_path)


# %%
path = "svd_50_2.txt"
def evaluate(path):
    print('[evaluate] Loading model...')
    model = load_model(path)

    print('[evaluate] Collecting matrix...')
    matrix, vocab, indices = collect(model)

    print('[evaluate] WordSim353 correlation:')
    ws = eval_wordsim(model)
    print(ws)

    print('[evaluate] BATS accuracies:')
    bats = eval_bats(model, matrix, vocab, indices)
    print(bats)

    print('[evaluate] MSR accuracy:')
    msr = eval_msr(model)
    print(msr)
    return ws,bats,msr


# %%
table = pd.DataFrame()


# %%
for win in [2,5,10]:
    for dim in [50,100,300]:
        path = f"svd_{dim}_{win}.txt"
        ws,bats,msr = evaluate(path)
        row = {"Algorithm":"SVD", "Win.":win, "Dim.":dim, "N. s.":"-", "WordSim":ws[0]*100, "BATS Male-Female":np.round(bats["E10 [male - female]"],2), "BATS hypernym - misc": np.round(bats["L02 [hypernyms - misc]"],2) , "BATS total":np.round(bats["total"],2), "MSR":msr}
        table = table.append(pd.DataFrame(row,index=[0]), ignore_index = True)


# %%
table.to_csv("svd_results.csv")


# %%



