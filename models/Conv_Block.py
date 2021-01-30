from tensorflow import keras

class Conv_Block(keras.Model):
  
  def __init__(self, filters, strides=1, kernel=3, padding = 'same', dropRate = 0.1):
    super(Conv_Block, self).__init__()

    self.convLayer = keras.layers.Conv2D(filters, kernel, strides=strides, padding=padding)
    self.bnLayer = keras.layers.BatchNormalization()
    self.actRelu = keras.layers.Activation('relu')
    self.dropout = keras.layers.Dropout(dropRate)
  
  def call(self, x, training=None):
    x = self.convLayer(x)
    x = self.bnLayer(x, training = training)
    x = self.actRelu(x)
    x = self.dropout(x)

    return x
