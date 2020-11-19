import tensorflow as tf
from tensorflow import keras

from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Flatten
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
!pip install -U git+https://github.com/qubvel/efficientnet

import efficientnet.keras as enet

from keras import backend as K
from keras.metrics import top_k_categorical_accuracy #for top1 top5
#top_values, top_indices = K.compat.v1.get_session().run(tf.nn.top_k(_pred_test, k=5))


import collections
import os
import math
import string

from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.utils import np_utils

from keras.models import Sequential
from keras.layers import Dense, Activation

import tensorflow as tf
import tensorflow_model_optimization as tfmot
quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer


from keras_applications.imagenet_utils import _obtain_input_shape
from keras_applications.imagenet_utils import preprocess_input as _preprocess_input
from keras import utils as keras_utils

from tensorflow.keras import layers
from keras.layers import Input
from six.moves import xrange

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import backend as K


# loading Cifar-10 dataset

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


# Swish defination
from keras.backend import sigmoid

class SwishActivation(Activation):
    
    def __init__(self, activation, **kwargs):
        super(SwishActivation, self).__init__(activation, **kwargs)
        self.__name__ = 'swish_act'

def swish_act(x, beta = 1):
    return (x * sigmoid(beta * x))

from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
get_custom_objects().update({'swish_act': SwishActivation(swish_act)})


def EfficientNet(width_coefficient,
                 depth_coefficient,
                 default_resolution,
                 dropout_rate=0.2,
                 drop_connect_rate=0.2,
                 depth_divisor=8,
                 blocks_args=DEFAULT_BLOCKS_ARGS,
                 model_name='efficientnet',
                 include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 classes=1000,
                 **kwargs):
  
    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if not (weights in {'imagenet', 'noisy-student', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=default_resolution,
                                      min_size=32,
                                      data_format=keras.backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = tf.keras.layers.Input(shape=input_shape)
        #img_input = Input(shape=input_shape)
    else:
        if backend.backend() == 'tensorflow':
            from tensorflow.python.keras.backend import is_keras_tensor
        else:
            is_keras_tensor = backend.is_keras_tensor
        if not is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3 if keras.backend.image_data_format() == 'channels_last' else 1
    activation = get_swish(**kwargs)

    
    # Build stem
    x = img_input
    
    x = tf.keras.layers.Conv2D(round_filters(32, width_coefficient, depth_divisor), 3,
                      strides=(2, 2),
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name='stem_conv')(x)
    '''
    x = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Conv2D(round_filters(32, width_coefficient, depth_divisor), 3,
                      strides=(2, 2),
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name='stem_conv'))(x)
                      '''
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name='stem_bn')(x)
    x = tf.keras.layers.Activation(activation, name='stem_activation')(x)
    #x = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Activation('relu', name='stem_activation'))(x)

    # Build blocks
    num_blocks_total = sum(block_args.num_repeat for block_args in blocks_args)
    block_num = 0
    for idx, block_args in enumerate(blocks_args):
        assert block_args.num_repeat > 0
        # Update block input and output filters based on depth multiplier.
        block_args = block_args._replace(
            input_filters=round_filters(block_args.input_filters,
                                        width_coefficient, depth_divisor),
            output_filters=round_filters(block_args.output_filters,
                                         width_coefficient, depth_divisor),
            num_repeat=round_repeats(block_args.num_repeat, depth_coefficient))

        # The first block needs to take care of stride and filter size increase.
        drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
        x = mb_conv_block(x, block_args,
                          activation=activation,
                          drop_rate=drop_rate,
                          prefix='block{}a_'.format(idx + 1))
        block_num += 1
        if block_args.num_repeat > 1:
            # pylint: disable=protected-access
            block_args = block_args._replace(
                input_filters=block_args.output_filters, strides=[1, 1])
            # pylint: enable=protected-access
            for bidx in xrange(block_args.num_repeat - 1):
                drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
                block_prefix = 'block{}{}_'.format(
                    idx + 1,
                    string.ascii_lowercase[bidx + 1]
                )
                x = mb_conv_block(x, block_args,
                                  activation=activation,
                                  drop_rate=drop_rate,
                                  prefix=block_prefix)
                block_num += 1

    # Build top
    x = tf.keras.layers.Conv2D(round_filters(1280, width_coefficient, depth_divisor), 1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name='top_conv')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name='top_bn')(x)
    x = tf.keras.layers.Activation(activation, name='top_activation')(x)
    #x = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Activation('relu', name='stem2_activation'))(x)
    
    if include_top:
        x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        if dropout_rate and dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name='top_dropout')(x)
        x = keras.layers.Dense(classes,
                         activation='softmax',
                         kernel_initializer=DENSE_KERNEL_INITIALIZER,
                         name='probs')(x)
    else:
        if pooling == 'avg':
            x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling2D(name='max_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = tf.keras.Model(inputs, x, name=model_name)

    # Load weights.
    if weights == 'imagenet':

        if include_top:
            file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_autoaugment.h5'
            file_hash = IMAGENET_WEIGHTS_HASHES[model_name][0]
        else:
            file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
            file_hash = IMAGENET_WEIGHTS_HASHES[model_name][1]
        weights_path = tf.keras.utils.get_file(
            file_name,
            IMAGENET_WEIGHTS_PATH + file_name,
            cache_subdir='models',
            file_hash=file_hash,
        )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model

def EfficientNetB7(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        **kwargs
):
    return EfficientNet(
        2.0, 3.1, 600, 0.5,
        model_name='efficientnet-b7',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, classes=classes,
        **kwargs
    )



# loading B7 pre-trained on ImageNet without final aka feature extractor
model = EfficientNetB7(include_top=False, input_shape=(32,32,3), pooling='avg', weights='imagenet')

# building 2 fully connected layer 
x = model.output


x = tf.keras.layers.BatchNormalization()(x)
x = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Dropout(0.7))(x)

#x = tf.keras.layers.Dense(512)(x)
x = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Dense(512))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Activation('relu'))(x)
x = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Dropout(0.5))(x)

x = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Dense(128))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Activation('relu'))(x)
'''
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.7)(x)

x = tf.keras.layers.Dense(512)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)

x = tf.keras.layers.Dense(128)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation('relu')(x)
'''
def acc_top5(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)

# output layer
predictions = tf.keras.layers.Dense(10, activation="softmax")(x)
#print(model.input)
model_final = tf.keras.Model(inputs = model.input, outputs = predictions)

model_final.summary()


# model compilation
model_final.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(0.0001),
              metrics=['accuracy', acc_top5])

mcp_save = tf.keras.callbacks.ModelCheckpoint('EnetB7_CIFAR10_TL.h5', save_best_only=True, monitor='val_acc')
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, verbose=1,)

#print("Training....")
model_final.fit(x_train, y_train,
              batch_size=32,
              epochs=10,
              validation_split=0.1,
              callbacks=[mcp_save, reduce_lr],
              shuffle=True,
              verbose=1)
              
              
quant_aware_model = tfmot.quantization.keras.quantize_apply(model_final)
quant_aware_model.summary()

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
# model compilation
'''
quant_aware_model.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.0001),
              metrics=['accuracy', acc_top5])
'''
quant_aware_model.compile(optimizer='adam',
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy',acc_top5])

mcp_save = ModelCheckpoint('EnetB8_CIFAR10_TL.h5', save_best_only=True, monitor='val_acc')
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, verbose=1,)

print("Training....")

quant_aware_model.fit(x_train, y_train,
              batch_size=32,
              epochs=10,
              validation_split=0.1,
              callbacks=[mcp_save, reduce_lr],
              shuffle=True,
              verbose=1)
'''
quant_aware_model.fit(x_train,  y_train,
                  batch_size=32, epochs=10, validation_split=0.1)'''

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
# model compilation
'''
quant_aware_model.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.0001),
              metrics=['accuracy', acc_top5])
'''
quant_aware_model.compile(optimizer='adam',
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy',acc_top5])

mcp_save = ModelCheckpoint('EnetB8_CIFAR10_TL.h5', save_best_only=True, monitor='val_acc')
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, verbose=1,)

print("Training....")

quant_aware_model.fit(x_train, y_train,
              batch_size=32,
              epochs=10,
              validation_split=0.1,
              callbacks=[mcp_save, reduce_lr],
              shuffle=True,
              verbose=1)
'''
quant_aware_model.fit(x_train,  y_train,
                  batch_size=32, epochs=10, validation_split=0.1)'''


_, acc,acc_top5 = quant_aware_model.evaluate(x_test, y_test)
print("Test Accuracy: {}%".format(acc*100))
print("Top 5 Accuracy: {}%".format(acc_top5*100))





