from data_preprocess import load_data
from models.Fac_Model import Fac_Model
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

X_train, X_test, X_val, y_train, y_test, y_val = load_data()

model = Fac_Model(7)
model.build(input_shape=(None, 48, 48, 1))
model.summary()
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam' ,
    metrics=['accuracy']
  )

reducePlateau = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, min_delta=0.0001, patience=1, verbose=1)
history = model.fit(X_train, y_train, epochs=14,  batch_size=128, steps_per_epoch=250, validation_data=(X_val, y_val), verbose=1, callbacks=[ reducePlateau ])
result = model.evaluate(X_test, y_test, verbose=1)
print('Resultado Final: ', result)

model.save_weights('./saved_weights.h5')