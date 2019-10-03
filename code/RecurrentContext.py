import keras.backend as K
from keras.layers import Layer
from keras import initializers, regularizers, constraints
import tensorflow as tf
from GRU import GRU

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

class RecurrentContext(Layer):
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

        self.GRU = GRU(self, 100, 100)

        super(RecurrentContext, self).__init__(**kwargs)

    def build(self, input_shape):
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

        super(RecurrentContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):

        #self.GRU.reset_h()

        def compute_att(res, x_):
            context = res[0]

            uit = dot_product(x_, self.W)
            c = dot_product(context, self.W_context)

            if self.bias:
                uit += self.b

            uit = K.tanh(tf.add(uit, K.expand_dims(c, 1)))

            ait = dot_product(uit, self.u)
            a = K.exp(ait)
            # apply mask after the exp. will be re-normalized next
            if mask is not None:
                # Cast the mask to floatX to avoid float64 upcasting in theano
                #a *= K.cast(mask, K.floatx())
                pass
            # in some cases especially in the early stages of training the sum may be almost zero
            # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
            a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
            a_ = K.expand_dims(a)
            weighted_input = x_ * a_
            attended = K.sum(weighted_input, axis=1)

            context = self.GRU.forward_pass(context, attended)

            return [context, a]

        x_t = tf.transpose(x, [1,0,2,3])

        output, weights = tf.scan(compute_att, x_t,
                                  initializer=[K.zeros_like(x_t[0,:,0,:]),
                                               K.zeros_like(x_t[0,:,:,0])])

        output = tf.transpose(output, [1,0,2])
        weights = tf.transpose(weights, [1,0,2]) 

        if self.return_coefficients:
            return [output, weights]

        else:
            return [outputs]

    def compute_output_shape(self, input_shape):
        if self.return_coefficients:
            return [(input_shape[0], input_shape[1], input_shape[-1]), 
                    (input_shape[0], input_shape[1], input_shape[-1], 1)]
        else: 
            return [(input_shape[0], input_shape[1], input_shape[-1])]
