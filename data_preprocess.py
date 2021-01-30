import pandas as pd
import numpy as np
from tensorflow import keras
from keras.utils import np_utils
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import cv2 as cv

def load_data():
    data = pd.read_csv('./data/fer2013.csv')
    array = []

    for line in tqdm(data['pixels']):
        img = np.array(list(map(lambda e: int(e), line.split(" "))), 'float32')
        img /= 255
        array.append(img.reshape(48, 48, 1))

    dataframe = np.asarray(array)

    X = dataframe
    y = data['emotion'].values

    labels = pd.DataFrame(y)
    y = np_utils.to_categorical(labels)

    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=.5, random_state=42, shuffle=True)

    return [X_train, X_test, X_val, y_train, y_test, y_val]

def webcam_img_process(img):
    img = cv.resize(img, (48, 48))
    img = np.array(list(img), 'float32')
    img = np.divide(img, 255, out=img, casting='unsafe')
    img = img.reshape(48, 48, 1)

    array = []
    array.append(img)
    array = np.asarray(array)

    return array