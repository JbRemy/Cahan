import errno
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from gensim.models import KeyedVectors

from keras import optimizers
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Embedding, Dropout, Dense, TimeDistributed

# = = = = 

def plot_results(my_results,loss_acc_lr,my_title):
    assert loss_acc_lr in ['loss','categorical_accuracy','lr'], 'invalid 2nd argument!'
    n_epochs = len(my_results['loss'])
    if loss_acc_lr in ['loss','categorical_accuracy']:
        plt.plot(range(1,n_epochs+1),my_results['val_' + loss_acc_lr],label='validation')
        plt.plot(range(1,n_epochs+1),my_results[loss_acc_lr],label='training')
    else:
        plt.plot(range(1,n_epochs+1),my_results['lr'],label='learning rate')
    plt.legend(loc=0)
    plt.xlabel('epochs')
    plt.ylabel(loss_acc_lr)
    plt.xlim([1,n_epochs+1])
    plt.grid(True)
    if loss_acc_lr == 'acc':
        plt.title("Best result %.2f" % (100*max(my_results['val_' + loss_acc_lr])))
    else:
        plt.title(my_title)

# = = = = 

dataset_name = 'yahoo_answers_csv' #'amazon_review_full_csv' #'yahoo_answers_csv' #'amazon_review_full_csv'

n_cats = 10 #5 #10 #5
is_GPU = True # False
n_units = 50
max_doc_size_overall = 20 # max nb of sentences allowed per document
max_sent_size_overall = 50 # max nb of words allowed per sentence
drop_rate = 0.5
my_loss = 'categorical_crossentropy'

n_runs = 1
nb_epochs_train = 50

my_prec = 5 # nb of decimals to keep in history files

runs = ['run%i' % i for i in range(n_runs)]

# replace with your own!
path_root = './data/' + dataset_name + '/'
path_to_batches = path_root + '/batches_' + dataset_name + '/'
model_path = "./V1/baseline/"
path_to_save =  model_path + dataset_name + '/'
path_to_functions = './'

path_to_weights = './V1/weights/'

# Creating files for results saving
try:
    os.mkdir(path_to_save)

except OSError as e:
    if e.errno == errno.EEXIST:
        pass
    else:
        os.mkdir(model_path)
        os.mkdir(path_to_save)

if os.path.exists(path_to_save + 'res.json'):
    with open(path_to_save + 'res.json', 'r', encoding='utf-8') as my_file:
        res = json.load(my_file)

else:
    res = {}

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
if not os.path.exists(path_to_save + "/init_weights"):
    han.save_weights(path_to_save + "/init_weights")

# LR test
max_lr, base_lr = 3, 0.001

# loading batches names
batch_names = os.listdir(path_to_batches)
batch_names_train = [elt for elt in batch_names if 'train_' in elt]
batch_names_val = [elt for elt in batch_names if 'val_' in elt]
its_per_epoch_train = int(len(batch_names_train)/2)
its_per_epoch_val = int(len(batch_names_val)/2)
n_its_plot = int(its_per_epoch_train/10)

half_cycle = 6
my_patience = int(half_cycle*2)
nb_epochs_lrtest = half_cycle

step_size = its_per_epoch_train*half_cycle # nb of iterations per half cycle

print("Half_cycle = ", half_cycle)

base_mt, max_mt = 0.85, 0.95

# LR TEST
if not "lr" in res:
    print('doing LR test')
    han.load_weights(path_to_save + "/init_weights")

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
                metrics=['categorical_accuracy'])    

    early_stopping = EarlyStopping(monitor='val_categorical_accuracy', # go through epochs as long as accuracy on validation set increases
                                   patience=3,
                                   mode='max')
    
    lr_sch = CyclicLR(base_lr=base_lr, max_lr=max_lr, step_size=step_size, mode='triangular')
    mt_sch = CyclicMT(base_mt=base_mt, max_mt=max_mt, step_size=step_size, mode='triangular')

    # batch-based callbacks
    loss_hist = LossHistory()
    acc_hist = AccHistory()
    val_batch_acc_hist = ValBatchAccHistory(rd_val, its_per_epoch_val,
                                            n_its_plot)
    lr_hist = LRHistory()
    mt_hist = MTHistory() 

    callback_list = [loss_hist,acc_hist,lr_hist,mt_hist,lr_sch,mt_sch,val_batch_acc_hist,early_stopping]

    # training
    print('STARTING LR TEST')
    han.fit_generator(rd_train, 
                      steps_per_epoch=its_per_epoch_train, 
                      epochs=nb_epochs_lrtest,
                      callbacks=callback_list,
                      validation_data=rd_val, 
                      validation_steps=its_per_epoch_val,
                      use_multiprocessing=False, 
                      workers=1)

    hist = han.history.history
    hist['batch_loss'] = loss_hist.loss_avg
    hist['batch_acc'] = acc_hist.acc_avg
    hist['batch_lr'] = lr_hist.lrs
    hist['batch_mt'] = mt_hist.mts
    hist['val_batch_acc'] = val_batch_acc_hist.accs

    hist_to_save = {k: [str(round(elt,my_prec)) for elt in v] for k, v in hist.items()}
    with open(path_to_save + 'lr_range_test_trainval_history.json', 'w') as my_file:
        json.dump(hist_to_save, my_file, sort_keys=False, indent=4)

    #with open(path_to_save + 'lr_range_test_trainval_history.json' , 'r', encoding='utf8') as my_file:
    #    hist = json.load(my_file)
    #hist = {k:list(map(float,v)) for k,v in hist.items()} # convert strings to floats

    acc_plot = [elt for idx,elt in enumerate(hist['val_batch_acc'],0)]
    best_idx, smooth_acc_plot = get_plateau(acc_plot)
    lr_plot = [elt for idx,elt in enumerate(hist['batch_lr'],0) if idx%n_its_plot==0]
            
    max_lr = float(lr_plot[best_idx])
    base_lr = round(max_lr/6,5)

    fig = plt.figure(figsize=(10,7))
    plt.plot(lr_plot,acc_plot)
    plt.plot(lr_plot,smooth_acc_plot)
    plt.xlabel('learning rate')
    plt.ylabel('training accuracy')
    plt.grid(True)
    plt.title('Learning Rate Range Test')
    plt.axvline(x=base_lr,label='base LR',c='C1',ls='dashed')
    plt.axvline(x=max_lr,label='max LR',c='C2',ls='dotted')
    plt.legend()
    plt.savefig(path_to_save + '/lr_test_fig.png')

    print('RESULT : ', base_lr, max_lr)
    res["lr"] = [base_lr, max_lr]
    with open(path_to_save + 'res.json', 'w') as my_file:
        json.dump(res, my_file, sort_keys=False, indent=4)

else:
    base_lr, max_lr = res["lr"]

#nb_epochs = nb_epochs_train
batch_names_train = [elt for elt in batch_names if 'train_' in elt or 'val_' in elt]
batch_names_val = [elt for elt in batch_names if 'test_' in elt]
its_per_epoch_train = int(len(batch_names_train)/2)
its_per_epoch_val = int(len(batch_names_val)/2)
step_size = its_per_epoch_train*half_cycle

runs_to_do = [_ for _ in runs if _ not in res]
for run in runs_to_do:
    print('starting_' + run)
    han.load_weights(path_to_save + "/init_weights")

    try:
        os.mkdir(path_to_save + run + '/')

    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
    
    rd_train = read_batches(batch_names_train, path_to_batches,
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
                metrics=['categorical_accuracy']) 

    lr_sch = CyclicLR(base_lr=base_lr, max_lr=max_lr, step_size=step_size, mode='triangular')
    mt_sch = CyclicMT(base_mt=base_mt, max_mt=max_mt, step_size=step_size, mode='triangular')

    early_stopping = EarlyStopping(monitor='val_categorical_accuracy', # go through epochs as long as accuracy on validation set increases
                                   patience=my_patience,
                                   mode='max')

    # make sure that the model corresponding to the best epoch is saved
    checkpointer = ModelCheckpoint(filepath=path_to_save + run + '/' + 'trained_weights',
                                   monitor='val_categorical_accuracy',
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
    callback_list = [loss_hist,acc_hist,lr_hist,mt_hist,lr_sch,mt_sch,early_stopping,checkpointer]

    # training
    print('Starting Training')
    han.fit_generator(rd_train, 
                      steps_per_epoch=its_per_epoch_train, 
                      epochs=nb_epochs_train,
                      callbacks=callback_list,
                      validation_data=rd_val, 
                      validation_steps=its_per_epoch_val,
                      use_multiprocessing=False, 
                      workers=1)

    # Saving the model
    histt = han.history.history
    print(histt.keys())
    print(len(lr_hist.lrs))
    histt['batch_loss'] = loss_hist.loss_avg
    histt['batch_acc'] = acc_hist.acc_avg
    histt['batch_lr'] = lr_hist.lrs
    histt['batch_mt'] = mt_hist.mts

    histt_to_save = {k: [str(round(elt,my_prec)) for elt in v] for k, v in histt.items()}
    with open(path_to_save + run + '/' + 'history.json', 'w') as my_file:
        json.dump(histt_to_save, my_file, sort_keys=False, indent=4)
    
    #with open(path_to_save + run + '/' + 'history.json' , 'r', encoding='utf8') as my_file:
    #   histt = json.load(my_file)
    # convert strings to floats
    #histt = {k:list(map(float,v)) for k,v in histt.items() if k!='pcacc'}

    lr_epoch = [elt for idx,elt in enumerate(histt['batch_lr'],1) if
                idx%its_per_epoch_train==0]
    print(len(lr_epoch))
    histt['lr'] = lr_epoch
            
    fig = plt.figure(figsize=(12,8))
    plot_results(histt,'categorical_accuracy','accuracy')
    for ep_idx,foo in enumerate(range(nb_epochs_train),1):
        if ep_idx % half_cycle == 0:
            plt.axvline(x=ep_idx, c='C2',ls='dashed')

    plt.savefig(path_to_save + run + '/learning_curve.png')
    
    res[run] = {'val_categorical_accuracy':100*max(histt['val_categorical_accuracy']),'best ep nb':histt['val_categorical_accuracy'].index(max(histt['val_categorical_accuracy']))+1} # 1-based index!
    with open(path_to_save + 'res.json', 'w') as my_file:
        json.dump(res, my_file, sort_keys=False, indent=4)

res['final'] = np.mean([res[run][0] for run in runs])
res['final_std'] = np.std([res[run][0] for run in runs])

with open(path_to_save + 'res.json', 'w') as my_file:
    json.dump(res, my_file, sort_keys=False, indent=4)
