import os
import errno
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

from shutil import copyfile

from time import strftime
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from nltk import sent_tokenize, word_tokenize

import tensorflow as tf
from keras.models import Model
from keras import optimizers, backend as K
from keras.backend.tensorflow_backend import _to_tensor
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Embedding, Dropout, Dense, TimeDistributed, Concatenate, Lambda

# custom classes and functions
path_to_functions = './'
sys.path.insert(0, path_to_functions)
from ContextAwareSelfAttentionWindow import ContextAwareSelfAttention
from AttentionWithContext import AttentionWithContext
from CyclicalLearningRate import CyclicLR
from CyclicalMomentum import CyclicMT
from han_my_functions import read_batches, bidir_gru, PerClassAccHistory, LossHistory, \
        AccHistory, LRHistory, MTHistory, ValBatchAccHistory,\
        InformationRegularizer, CustomLossWrapper
from plateau import get_plateau

from sklearn.metrics import classification_report, confusion_matrix

sess = tf.Session()
K.set_session(sess)

dataset_name = 'yelp_review_full_csv'#'yahoo_answers_csv'##'amazon_review_full_csv'# #

n_cats = 5 #10 #5
is_GPU = False # False
n_units = 50
max_doc_size_overall = 20 # max nb of sentences allowed per document
max_sent_size_overall = 50 # max nb of words allowed per sentence
drop_rate = 0.5
my_loss = 'categorical_crossentropy'

# replace with your own!
path_root = './data/' + dataset_name + '/'
path_to_batches = path_root + '/batches_' + dataset_name + '/'
model_path = "./models/antoine/baseline/" + dataset_name 
path_to_save = './'
path_to_functions = './'

#path_to_weights = './models/jb/weights/' + dataset_name + '/'
path_to_weights = model_path
n_runs = 4
nb_epochs_train = 150

my_prec = 5 # nb of decimals to keep in history files

runs = ['run%i' % i for i in range(n_runs)]

# Loading vectors
gensim_obj = KeyedVectors.load(path_root + 'word_vectors.kv', mmap='r') # needs an absolute path!
word_vecs = gensim_obj.wv.syn0
# add Gaussian initialized vector on top of embedding matrix (for padding)
pad_vec = np.random.normal(size=word_vecs.shape[1]) 
word_vecs = np.insert(word_vecs,0,pad_vec,0)

# Defining Network
## Inputs
sent_ints = Input(shape=(None,))
sent_wv = Embedding(input_dim=word_vecs.shape[0],
                    output_dim=word_vecs.shape[1],
                    weights=[word_vecs],
                    input_length=None, # sentence size vary from batch to batch
                    trainable=True
                    )(sent_ints)

## Sentences encoder
sent_wv_dr = Dropout(drop_rate)(sent_wv)
sent_wa = bidir_gru(sent_wv_dr,n_units,is_GPU) # annotations for each word in the sentence
sent_att_vec,sent_att_coeffs = AttentionWithContext(return_coefficients=True)(sent_wa) # attentional vector for the sentence
sent_att_vec_dr = Dropout(drop_rate)(sent_att_vec)                      
sent_encoder = Model(sent_ints,sent_att_vec_dr)
sent_encoder_weights = Model(sent_ints,sent_att_coeffs)

## Documents Encoder
doc_ints = Input(shape=(None,None,))        
sent_att_vecs_dr = TimeDistributed(sent_encoder)(doc_ints)
doc_sa = bidir_gru(sent_att_vecs_dr,n_units,is_GPU) # annotations for each sentence in the document
doc_att_vec, doc_att_coeffs = AttentionWithContext(return_coefficients=True)(doc_sa) # attentional vector for the document
doc_att_vec_dr = Dropout(drop_rate)(doc_att_vec)

## Output
preds = Dense(units=n_cats,
              activation='softmax')(doc_att_vec_dr)

han = Model(doc_ints,preds)
my_optimizer = optimizers.SGD(lr=0.1,
                              momentum=0.1)
han.compile(loss=my_loss,
                optimizer=my_optimizer,
                metrics=['categorical_accuracy'])    

for run in runs:

    han.load_weights(model_path +"/"+run+"/"+"trained_weights")

    batch_names = os.listdir(path_to_batches)
    batch_names_val = [elt for elt in batch_names if 'test_' in elt]
    its_per_epoch_val = int(len(batch_names_val)/2)

    rd_val = read_batches(batch_names_val,
                      path_to_batches,
                      do_shuffle=False,
                      do_train=True,
                      my_max_doc_size_overall=max_doc_size_overall,
                      my_max_sent_size_overall=max_sent_size_overall,
                      my_n_cats=n_cats)

    print(han.evaluate_generator(rd_val, steps=its_per_epoch_val, verbose=1))

