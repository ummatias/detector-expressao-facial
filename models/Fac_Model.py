from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras import backend
from models.Conv_Block import Conv_Block

def swish_ann_activation_function(x):
  return backend.sigmoid(x) * x

class Fac_Model(keras.Model):
    def __init__(self, classNum, **kwargs):
        super(Fac_Model, self).__init__(**kwargs)

        self.block1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(1,48,48))
        self.block2 = Conv_Block(32)
        self.pollingM1 = keras.layers.MaxPooling2D(pool_size=(2, 2))

        self.block3 = Conv_Block(64)
        self.block4 = Conv_Block(64)
        self.pollingM2 = keras.layers.MaxPool2D(pool_size=(2, 2))

        self.block5 = Conv_Block(96)
        self.block6 = Conv_Block(96, padding='valid')
        self.pollingM3 = keras.layers.MaxPool2D(pool_size=(2, 2))

        self.block7 = Conv_Block(128)
        self.block8 = Conv_Block(128, padding='valid')
        self.pollingM4 = keras.layers.MaxPool2D(pool_size=(2, 2))

        self.flatten = keras.layers.Flatten()
        self.dense = keras.layers.Dense(64, activation=swish_ann_activation_function)
        self.drop = keras.layers.Dropout(0.4)
        self.outputLayer = keras.layers.Dense(classNum, activation='sigmoid')  

    def call(self, inputs,training=None):

        x = inputs

        x = self.block1(x)
        x = self.block2(x)
        x = self.pollingM1(x)

        x = self.block3(x)
        x = self.block4(x)  
        x = self.pollingM2(x) 

        x = self.block5(x)
        x = self.block6(x)  
        x = self.pollingM3(x) 

        x = self.block7(x)
        x = self.block8(x)
        x = self.pollingM4(x)

        x = self.flatten(x)
        x = self.dense(x)
        x = self.drop(x)    
        x = self.outputLayer(x)     

        return x   
