from keras.layers import Input, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from tensorflow import Tensor

from typing import Tuple, Union

def network(imshape: Union[Tuple[int, int], int], depth: int = 17) -> Tuple[Tensor, Tensor]:
    if isinstance(imshape, int):
        imshape = (imshape, imshape)
    input = Input(imshape + (1,))
    
    tensor = convolutional(64, 3)(input)
    tensor = Activation('relu')(tensor)
    for x in range(depth - 2):
        tensor = convolutional(64, 3)(tensor) 
        tensor = BatchNormalization(axis=-1)(tensor)
        tensor = Activation('relu')(tensor)
    output = convolutional(1, 3)(tensor) 
    return input, output

def convolutional(filters, kernel_size):
    return Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size),
                  padding="same",
                  kernel_initializer="he_normal",
                  data_format="channels_last",
                  dilation_rate=1,
                  strides=1,                  
                 )