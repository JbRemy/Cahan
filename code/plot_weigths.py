import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import errno
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from gensim.models import KeyedVectors

from keras import optimizers, backend as K
from keras import optimizers
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Embedding, Dropout, Dense, TimeDistributed

from keras.backend.tensorflow_backend import _to_tensor

from nltk import sent_tokenize, word_tokenize

sess = tf.Session()
K.set_session(sess)

dataset_name = 'amazon_review_full_csv'#'yelp_review_full_csv'# #'yahoo_answers_csv' #'amazon_review_full_csv' #'yahoo_answers_csv' #'amazon_review_full_csv'

n_cats = 5 #10 #5 #10 #5
is_GPU = False # False
n_units = 50
max_doc_size_overall = 20 # max nb of sentences allowed per document
max_sent_size_overall = 50 # max nb of words allowed per sentence
drop_rate = 0.5
my_loss = 'categorical_crossentropy'

n_runs = 4
nb_epochs_train = 150

my_prec = 5 # nb of decimals to keep in history files

runs = ['run%i' % i for i in range(n_runs)]

# replace with your own!
path_root = './data/' + dataset_name + '/'
path_to_batches = path_root + '/batches_' + dataset_name + '/'
model_path = "./V1/baseline/" + dataset_name + "/run1/"
path_to_save = './'
path_to_functions = './'

path_to_weights = './V1/weights/'

# custom classes and functions
sys.path.insert(0, path_to_functions)
from AttentionWithContext import AttentionWithContext
from CyclicalLearningRate import CyclicLR
from CyclicalMomentum import CyclicMT
from han_my_functions import read_batches, bidir_gru, LossHistory, \
        AccHistory, LRHistory, MTHistory, ValBatchAccHistory
from plateau import get_plateau

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

han.load_weights(model_path+"trained_weights")

with open(path_root + 'vocab.json', 'r') as my_file:
    word_to_index_han = json.load(my_file)
            
padding_idx = 0
oov_idx = 1

# ===== toy example =====
text = "terrible value .  ordered pasta entree .  .  $ 16.95 good taste but size was an appetizer size .  .  no salad , no bread no vegetable .  this was .  our and tasty cocktails .  our second visit .  i will not go back ."
#text ='The cocktail and pasta entree that we ordered were tasty, but size was small. The cocktail and pasta entree that we ordered were tasty, but size was small. The cocktail and pasta entree that we ordered were tasty, but size was small.'

get_sent_att_coeffs = Model(sent_ints,sent_att_coeffs) # coeffs over the words
get_doc_attention_coeffs = Model(doc_ints,doc_att_coeffs) # coeffs over the documents

# == preprocessing ==
sents = sent_tokenize(text)
sents_idxs = []
sents_tokenized = []
for sent in sents:
    words = word_tokenize(sent)
    sents_tokenized.append(words)
    idxs = [word_to_index_han[elt] if elt in word_to_index_han else oov_idx for elt in words]
    sents_idxs.append(idxs)

max_sent_size = min(max([len(s) for s in sents_idxs]),max_sent_size_overall)
sents_idxs_padded = [s+[padding_idx]*(max_sent_size-len(s)) if len(s)<max_sent_size else s[:max_sent_size] for s in sents_idxs]
reshaped_sentences = np.reshape(np.array(sents_idxs_padded),(1,len(sents),max_sent_size))
reshaped_sentences_tensor = _to_tensor(reshaped_sentences,dtype='float32') # a layer, unlike a model, requires tf tensor as input

print('== attention over words ==')
sents_att_coeffs = TimeDistributed(get_sent_att_coeffs)(reshaped_sentences_tensor)
word_coeffs = sents_att_coeffs.eval(session=sess)
word_coeffs = np.reshape(word_coeffs,(len(sents),max_sent_size))
doc_att_tensor = get_doc_attention_coeffs(reshaped_sentences_tensor)
doc_att = doc_att_tensor.eval(session=sess)[0]
res_tensor = han(reshaped_sentences_tensor)
res = res_tensor.eval(session=sess)
print(doc_att)

my_wcs = []
my_values_array = []
my_keys_array = []
for my_idx,wc in enumerate(word_coeffs):
    my_keys = sents_tokenized[my_idx]
    my_values = [round(elt,2) for elt in wc.tolist()[:len(my_keys)]]
    my_values_array.append(my_values[:len(my_values)])
    my_keys_array.append(my_keys)
    my_wcs.append(list(zip(my_keys,my_values)))

max_len = max([len(_) for _ in my_values_array])
my_values_array = np.array([_+[0]*(max_len-len(_)) for _ in my_values_array])
my_values_array /= np.max(my_values_array) 

print(my_wcs)

colors = ["#ffffff","#fbe2e2","#f8c6c6","#f5aaaa","#f28d8d","#ef7171","#ec5555","#e93838","#e61c1c","#e30000"]
colors_green = ["#ffffff","#e2ede3","#c6dbc8","#aac9ad","#8db792","#71a577","#55935c","#388141","#1c6f26","#005e0b"]
res_colors = ["#0013ff","#3817c8","#711c92","#aa205b","#e32525"]
res_str = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
html_code = ''

import colored
for i, keys in enumerate(my_keys_array):
    to_print = ''
    html_code += '<span style="background-color: %s;color: %s">---</span>' % (colors_green[int(9*doc_att[i])], colors_green[int(4*doc_att[i])])
    for j, key in enumerate(keys):
        html_code += '<span style="background-color: %s"> %s </span>' % (colors[int(9*my_values_array[i,j])], key)
        to_print += ' %s%s%s%s ' % (colored.fg('white'), colored.bg(232 + 2*int(my_values_array[i,j]*10)), key, colored.attr('reset'))
    print(to_print)
    html_code += '<br>\n'
html_code += '<span style="background-color: %s;color: %s">%s</span>' % (res_colors[np.argmax(res)], "#ffffff", res_str[np.argmax(res)])
print(html_code)
