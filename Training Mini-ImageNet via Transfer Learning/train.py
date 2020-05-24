
# Import Libraries
import sys; import numpy as np; import keras;
from keras import layers, regularizers;
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Conv2D;
from keras.layers import BatchNormalization, ZeroPadding2D, AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D;
from keras.models import Model; from keras.utils import layer_utils;
from keras.initializers import glorot_uniform; from keras.applications.resnet50 import ResNet50, preprocess_input;
from keras.preprocessing.image import ImageDataGenerator;
from keras.optimizers import SGD, Adam;
import keras.backend as K;

# Set Format
K.set_image_data_format('channels_last')

# Inputs
train_dir = sys.argv[1]
model_save = sys.argv[2]

# Load the Pre-Trained Model
resModel = ResNet50(weights = 'imagenet', include_top = False, input_shape = (256, 256, 3))

# Process and Generate Train Data
datagen_train = ImageDataGenerator(preprocessing_function = preprocess_input)
generate_train = datagen_train.flow_from_directory(train_dir, target_size = (256, 256), batch_size = 8)

# Transfer Learning New Model
def tmodel(resModel, flayers, num_classes):
    for layer in resModel.layers:
        layer.trainable = False
    
    X = resModel.output
    X = Flatten()(X)
    for l in flayers:
        X = Dense(l, activation = 'relu')(X)
        X = Dropout(0.5)(X)
    
    preds = Dense(num_classes, activation = 'softmax')(X)
    model_fin = Model(inputs = resModel.input, outputs = preds)
    
    return model_fin

# Initiate & Train the Model
model_fin = tmodel(resModel, flayers = [4096,4096], num_classes = 10)
epochs = 2
batchSize = 8
num_images = 12739

adm = Adam(lr = 0.00001)
model_fin.compile(adm, loss = 'categorical_crossentropy', metrics =  ['accuracy'])
model_fin.fit_generator(generate_train, epochs = epochs, steps_per_epoch = num_images // batchSize, shuffle = True)

model_fin.save(model_save)
