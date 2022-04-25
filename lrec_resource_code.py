'''
Resource code for:
"TBD3: A Thresholding-Based Dynamic Depression Detection from Social Media for Low-Resource Users" - LREC 2022
Authors: Hrishikesh Kulkarni, Sean MacAvaney, Nazli Goharian, Ophir Frieder
Georgetown University, Washington DC, USA

Code inspired from:
"Depression and Self-Harm Risk Assessment in Online Forums" - EMNLP 2017
Authors: Andrew Yates, Arman Cohan, Nazli Goharian
Georgetown University, Washington DC, USA
'''

import gzip
import json
import numpy as np
import os
import pickle
import nltk
import re
import heapq
import time
import sys
import random
import sklearn.metrics
from sklearn.metrics import precision_recall_curve
import argparse
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input, TimeDistributed, Activation, Masking, Convolution1D, \
    MaxPooling1D, Flatten, AveragePooling1D, GlobalAveragePooling1D
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer
random.seed(1234)

POST_LEN = 2000 #max number of posts for training

def datagen(number_of_batches, nn_max_posts_test, max_length=100, stype='training',
            batch_size=32, force_full=False, mintf=1, mindf=2, percentage=1.0, median = False, part='first'):
    assert stype in ['training', 'validation', 'testing']
    fn = '../../data/%s.gz' % stype
    print("loading %s posts" % stype)
    f = gzip.open(fn, 'rt')
    looponce = force_full or stype == 'testing'

    labels = {}
    allposts = {}
    no_of_users = number_of_batches * batch_size
    user_ctr = 0
    for i, line in enumerate(f):

        if user_ctr == no_of_users:
            print(str(len(allposts)) + ' users data loaded')
            f.close()
            break

        user_ctr += 1
        user = str(i)
        d = json.loads(line)[0]
        if median == 'True':
            limit = 645
        else:
            limit = 0


        if stype == 'training' or len([post for dt, post in d['posts']]) > limit: #645
            ###
            if d['label'] == 'control':
                labels[user] = np.array([1, 0], dtype=np.float32)
            elif d['label'] == 'depression':
                labels[user] = np.array([0, 1], dtype=np.float32)
            elif d['label'] is None:
                continue
            else:
                raise RuntimeError("unknown label: %s" % d['label'])

            post_list = [post for dt, post in d['posts']]
            if stype == 'testing' or stype == 'validation':
                if part == 'first':
                    if percentage == 1.0:
                        post_list = post_list[:nn_max_posts_test]
                    else:
                        post_list = post_list[:int(percentage*len(post_list))]
                elif part == 'last':
                    if percentage == 1.0:
                        post_list = post_list[len(post_list)-nn_max_posts_test:]
                    else:
                        post_list = post_list[len(post_list)-int(percentage * len(post_list)):]
                elif part == 'random':
                    if percentage == 1.0:
                        post_list = [post_list[i] for i in sorted(random.sample([i for i in range(len(post_list))], nn_max_posts_test))]
                    else:
                        post_list = [post_list[i] for i in
                                     sorted(random.sample([i for i in range(len(post_list))], int(percentage*len(post_list))))]

                else:
                    print('ERROR')
                    exit()
            allposts[user] = post_list

    f.close()

    tokfn = "cnn_tok_tf%s_df%s.p" % (mintf, mindf)
    load_tokenizer = os.path.exists(tokfn)

    if load_tokenizer:
        print("loading tokenizer")
        tok = pickle.load(open(tokfn, 'rb'))
    else:
        assert stype == 'training', "cannot fit tokenizer on validation or testing data"
        print("tokenizing %s users" % len(allposts))
        tok = Tokenizer(num_words=None)
        tok.fit_on_texts(post for uposts in allposts.values() for post in uposts)

        # remove all tokens with a low DF or TF
        removed = 0
        for term in list(tok.word_index.keys()):
            if tok.word_docs[term] < mindf or tok.word_counts[term] < mintf:
                removed += 1
                del tok.word_docs[term]
                del tok.word_counts[term]
                del tok.word_index[term]
        tok.index_docs = None
        idxs = {}
        nexti = 1
        for term, oldi in sorted(tok.word_index.items()):
            idxs[term] = nexti
            nexti += 1
        assert len(tok.word_index) == len(idxs)
        tok.word_index = idxs

        print("terms removed: %s; remaining: %s" % (removed, len(tok.word_index)))
        pickle.dump(tok, open(tokfn, 'wb'), protocol=-1)

    nb_words = len(tok.word_index) + 1

    print("found %s words; generator ready" % nb_words)

    def vecify(uposts):
        if len(uposts) > POST_LEN:
            chosen = [uposts[i] for i in sorted(random.sample([i for i in range(len(uposts))], POST_LEN))]
        else:
            chosen = uposts

        seqs = pad_sequences(tok.texts_to_sequences(chosen), maxlen=max_length)
        if len(seqs) < POST_LEN:
            seqs = np.pad(seqs, ((0, POST_LEN - len(seqs)), (0, 0)), mode='constant')
        return seqs

    def gen():
        X, y = [], []

        while True:
            iterval = 0
            for user, uposts in allposts.items():
                iterval += 1
                X.append(vecify(uposts))
                y.append(labels[user])

                if len(X) == batch_size:
                    X, y = np.array(X), np.array(y)
                    yield (X.reshape(X.shape[0], X.shape[1] * X.shape[2]), y)

                    X, y = [], []

            if looponce and len(X) > 0:
                X, y = np.array(X), np.array(y)
                yield (X.reshape(X.shape[0], X.shape[1] * X.shape[2]), y)

                X, y = [], []

            if looponce:
                break

    return nb_words, gen

def main():
    print(tf.config.experimental.list_physical_devices('GPU'))
    args = argparse.ArgumentParser(description='Program description.')
    args.add_argument('-p', '--percentage', default=1.0, type=float, help='From 0 to 1')
    args.add_argument('-b', '--batch_size', default=32, type=int, help='Batch size')
    args.add_argument('-nob', '--no_of_batches', default=24, type=int, help='No of Batches')#1190
    args.add_argument('-mp', '--max_posts', default=2000, type=int, help='Max Posts')
    args.add_argument('-pt', '--part', default='first', type=str, help='first or last or random')
    args.add_argument('-m', '--model', default='cnn', type=str, help='ML algorithm')
    args.add_argument('-md','--median', default='False', type=str, help='True or False')
    args = args.parse_args()

    print('Argparse output:', args)
    print('max posts: %d' % args.max_posts)


    # Testing a trained model
    TEST_SIZE = args.batch_size * args.no_of_batches
    # Generator for filtered data as per command line arguments
    nb_words, genf = datagen(number_of_batches=1, nn_max_posts_test=args.max_posts,
                             max_length=100, batch_size=TEST_SIZE, stype='validation',
                             force_full=True, percentage=args.percentage, median=args.median, part=args.part)


if __name__ == '__main__':
    main()
