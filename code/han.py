
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from time import strftime
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from nltk import sent_tokenize, word_tokenize

import tensorflow as tf
from keras.models import Model
from keras import optimizers, backend as K
from keras.backend.tensorflow_backend import _to_tensor
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Embedding, Dropout, Dense, TimeDistributed

dataset_name = 'amazon_review_full_csv'
n_cats = 5 
is_GPU = True # False
n_units = 50
max_doc_size_overall = 20 # max nb of sentences allowed per document
max_sent_size_overall = 50 # max nb of words allowed per sentence
drop_rate = 0.45
my_loss = 'categorical_crossentropy'

# replace with your own!
path_root = './data/' + dataset_name + '/'
path_to_batches = path_root + '/batches_' + dataset_name + '/'
path_to_save = './models/HAN/results/' + dataset_name + '/'
path_to_functions = './'

# custom classes and functions
sys.path.insert(0, path_to_functions)
from AttentionWithContext import AttentionWithContext
from CyclicalLearningRate import CyclicLR
from CyclicalMomentum import CyclicMT
from han_my_functions import read_batches, bidir_gru, PerClassAccHistory, LossHistory, \
        AccHistory, LRHistory, MTHistory


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

## Documents Encoder
doc_ints = Input(shape=(None,None,))        
sent_att_vecs_dr = TimeDistributed(sent_encoder)(doc_ints)
doc_sa = bidir_gru(sent_att_vecs_dr,n_units,is_GPU) # annotations for each sentence in the document
doc_att_vec,doc_att_coeffs = AttentionWithContext(return_coefficients=True)(doc_sa) # attentional vector for the document
doc_att_vec_dr = Dropout(drop_rate)(doc_att_vec)

## Output
preds = Dense(units=n_cats,
              activation='softmax')(doc_att_vec_dr)

han = Model(doc_ints,preds)
# so that we can just load the initial weights instead of redifining the model later on
han.save_weights(path_to_save + 'han_init_weights')

# TRAINING

max_lr = 1.1
base_lr = round(max_lr/6, 5)
base_mt, max_mt = 0.85, 0.95

# loading batches names
batch_names = os.listdir(path_to_batches)
batch_names_train = [elt for elt in batch_names if 'train_' in elt or 'val_' in elt]
batch_names_val = [elt for elt in batch_names if 'test_' in elt]

nb_epochs = 50
half_cycle = 6 
my_patience = half_cycle*2
its_per_epoch_train = int(len(batch_names_train)/2)
its_per_epoch_val = int(len(batch_names_val)/2)
step_size = its_per_epoch_train*half_cycle

print(its_per_epoch_train,its_per_epoch_val,step_size)

rd_train = read_batches(batch_names_train,
                        path_to_batches,
                        do_shuffle=True,
                        do_train=True,
                        my_max_doc_size_overall=max_doc_size_overall,
                        my_max_sent_size_overall=max_sent_size_overall,
                        my_n_cats=n_cats)

rd_val = read_batches(batch_names_val,
                      path_to_batches,
                      do_shuffle=False,
                      do_train=True,
                      my_max_doc_size_overall=max_doc_size_overall,
                      my_max_sent_size_overall=max_sent_size_overall,
                      my_n_cats=n_cats)

my_optimizer = optimizers.SGD(lr=base_lr,
                              momentum=max_mt, # we decrease momentum when lr increases
                              decay=1e-5,
                              nesterov=True)

han.compile(loss=my_loss,
            optimizer=my_optimizer,
            metrics=['accuracy'])    

lr_sch = CyclicLR(base_lr=base_lr, max_lr=max_lr, step_size=step_size, mode='triangular')
mt_sch = CyclicMT(base_mt=base_mt, max_mt=max_mt, step_size=step_size, mode='triangular')

early_stopping = EarlyStopping(monitor='val_acc', # go through epochs as long as accuracy on validation set increases
                               patience=my_patience,
                               mode='max')

# make sure that the model corresponding to the best epoch is saved
checkpointer = ModelCheckpoint(filepath=path_to_save + 'han_trained_weights',
                               monitor='val_acc',
                               save_best_only=True,
                               mode='max',
                               verbose=0,
                               save_weights_only=True) # so that we can train on GPU and load on CPU (for CUDNN GRU)

# batch-based callbacks
loss_hist = LossHistory()
acc_hist = AccHistory()
lr_hist = LRHistory()
mt_hist = MTHistory() 

# epoch-based callbacks
pcacc_hist = PerClassAccHistory(my_n_cats=n_cats, my_rd=rd_val, my_n_steps=its_per_epoch_val)
callback_list = [loss_hist,acc_hist,lr_hist,mt_hist,lr_sch,mt_sch,early_stopping,checkpointer,pcacc_hist]

# training
han.fit_generator(rd_train, 
                  steps_per_epoch=its_per_epoch_train, 
                  epochs=nb_epochs,
                  callbacks=callback_list,
                  validation_data=rd_val, 
                  validation_steps=its_per_epoch_val,
                  use_multiprocessing=False, 
                  workers=1)

# Saving the model
hist = han.history.history
print(hist.keys())
print(len(lr_hist.lrs))
hist['batch_loss'] = loss_hist.loss_avg
hist['batch_acc'] = acc_hist.acc_avg
hist['batch_lr'] = lr_hist.lrs
hist['batch_mt'] = mt_hist.mts
hist['pcacc'] = pcacc_hist.per_class_accuracy

hist = {k: [str(elt) for elt in v] for k, v in hist.items()}
with open(path_to_save + 'han_history.json', 'w') as my_file:
    json.dump(hist, my_file, sort_keys=False, indent=4)
