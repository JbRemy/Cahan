import numpy as np
import keras.backend as K
from keras.layers import Layer, Dropout
from keras import initializers, regularizers, constraints
import tensorflow as tf

drop_rate = 0.5

def dot_product(x, kernel):
    """
    https://github.com/richliao/textClassifier/issues/13#issuecomment-377323318
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class ContextAwareSelfAttention(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.  by using a windowed context vector to assist the attention
    # Input shape
        4D tensor with shape: `(samples, sentence, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, sentence, features)`.

    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.

    Note: The layer has been tested with Keras 2.0.6

    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(ContextAwareSelfAttentionWindow())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self, context_version="classic", cut_gradient=False, aggregate="sum", discount=1, return_coefficients=False,
                 W_regularizer=None, W_context_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, W_context_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.context_version = context_version
        self.cut_gradient = cut_gradient
        self.aggregate = aggregate
        self.discount = discount
        self.supports_masking = True
        self.return_coefficients = return_coefficients
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.W_context_regularizer = regularizers.get(W_context_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.W_context_constraint = constraints.get(W_context_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(ContextAwareSelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        print(input_shape)
        assert len(input_shape) == 4

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)

        self.W_context = self.add_weight((input_shape[-1], input_shape[-1],),
                                         initializer=self.init,
                                         name='{}_W_context'.format(self.name),
                                         regularizer=self.W_context_regularizer,
                                         constraint=self.W_context_constraint)

        if self.context_version == "gate":
            self.W_l = self.add_weight((input_shape[-1],),
                                             initializer=self.init,
                                             name='{}_W_l'.format(self.name),
                                             regularizer=self.W_context_regularizer,
                                             constraint=self.W_context_constraint)

            self.W_lc = self.add_weight((input_shape[-1],),
                                             initializer=self.init,
                                             name='{}_W_lc'.format(self.name),
                                             regularizer=self.W_context_regularizer,
                                             constraint=self.W_context_constraint)

            self.bl = self.add_weight((1,),
                                     initializer='zero',
                                     name='{}_b_l'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(ContextAwareSelfAttention, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        def compute_att(res, x_):
            context = res[2]
            count = res[3]

            if self.aggregate == "mean":
                context_ = tf.cond(tf.less(count,1), lambda:context/(count+1.0), lambda: context/count)

            elif self.aggregate == "sum":
                context_ = context

            if self.cut_gradient:
                context_ = K.stop_gradient(context_)

            uit = dot_product(x_, self.W)
            c = dot_product(context_, self.W_context)

            if self.context_version == "classical":
                uit = K.tanh(tf.add(uit, K.expand_dims(c, 1)) + self.b)

            elif self.context_version == "gate":
                l = K.expand_dims(K.sigmoid(K.expand_dims(dot_product(context_, self.W_lc) +
                                           self.bl, 1)
                              + dot_product(x_, self.W_l)), -1)
                uit = K.tanh(tf.add(l*uit, (1-l)*K.expand_dims(c, 1)) + self.b)

            else:
                uit = K.tanh(uit)

            ait = dot_product(uit, self.u)

            a = K.exp(ait)

            # apply mask after the exp. will be re-normalized next
            if mask is not None:
                # Cast the mask to floatX to avoid float64 upcasting in theano
                a *= K.cast(mask, K.floatx())

            # in some cases especially in the early stages of training the sum may be almost zero
            # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
            a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

            a_ = K.expand_dims(a)
            weighted_input = x_ * a_

            attended = K.sum(weighted_input, axis=1)

            context *= self.discount
            context += attended

            return [attended, a, context, count+1]

        x_t = tf.transpose(x, [1,0,2,3])

        output, weights, context, _ = tf.scan(compute_att, 
                                     x_t,
                                     initializer=[K.zeros_like(x_t[0,:,0,:]),
                                                  K.zeros_like(x_t[0,:,:,0]),
                                                  K.zeros_like(x_t[0,:,0,:]),
                                                  0.0])

        output = tf.transpose(output, [1,0,2])
        context = tf.transpose(context, [1,0,2])
        weights = tf.transpose(weights, [1,0,2]) 

        try: 
            if False:
                sess = K.get_session()
                print(tf.norm(context, axis=[0,2]).eval(session=sess))
        except:
            pass

        if self.return_coefficients:
            return [output, context, weights]

        else:
            return [outputs, context]

    def compute_output_shape(self, input_shape):
        if self.return_coefficients:
            return [(input_shape[0], input_shape[1], input_shape[-1]), 
                    (input_shape[0], input_shape[1], input_shape[-1]), 
                    (input_shape[0], input_shape[1], input_shape[-1], 1)]
        else: 
            return [(input_shape[0], input_shape[1], input_shape[-1]),
                    (input_shape[0], input_shape[1], input_shape[-1])]
