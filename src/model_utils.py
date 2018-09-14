import numpy as np
import pandas as pd
import json
import csv

import tensorflow as tf
import keras.backend as K

from os import listdir, rename, makedirs
from os.path import isfile, join, exists
from shutil import copyfile

from keras.applications import densenet, xception
from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, EarlyStopping
from keras.optimizers import SGD, Adam
from keras.layers import Dropout, Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras import utils
from keras.preprocessing import image


img_height = 299
img_width = 299

batch_size = 16

def get_checkpoint(name):
    return f'checkpoints/{name}.hdf5'

def evaluate_model(model, tta_level, val_data_dir, preprocess_input_soft):
    test_datagen = image.ImageDataGenerator(preprocessing_function=xception.preprocess_input)

    validation_generator = test_datagen.flow_from_directory(
            val_data_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False)

    probs = []
    probs.append(model.predict_generator(validation_generator))

    for i in range(tta_level):
        test_datagen = image.ImageDataGenerator(preprocessing_function=preprocess_input_soft)

        validation_generator = test_datagen.flow_from_directory(
            val_data_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False)

        probs.append(model.predict_generator(validation_generator))
        
    return probs