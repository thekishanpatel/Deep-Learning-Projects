''' Kishan Patel '''

import os; import sys
import keras
from keras import layers
from keras.layers import Dense, Flatten
from keras.layers import Dropout
from keras.models import Model
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import keras.backend as K
K.set_image_data_format('channels_last')
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

train_path = sys.argv[1]
classes = 0
for dirs in os.walk(train_path):
        classes += 1
classes = classes - 1

tmodel = VGG19(weights = 'imagenet', include_top = False, input_shape = (256, 256, 3))

train_data_generator = ImageDataGenerator(preprocessing_function = preprocess_input)
train_generator = train_data_generator.flow_from_directory(train_path, target_size = (256, 256), batch_size = 8)

def mmodel(tmodel, f_layers, nclasses):
    for layer in tmodel.layers:
        layer.trainable = False
    
    X = tmodel.output
    X = Flatten()(X)
    for f in f_layers:
        X = Dense(f, activation = 'relu')(X)
        X = Dropout(0.5)(X)
    
    preds = Dense(nclasses, activation = 'softmax')(X)
    final_model = Model(inputs = tmodel.input, outputs = preds)
    
    return final_model

# Train Models
nepochs = 10
bsizes = 8
adam = Adam(lr = 0.00001)

final_model = mmodel(tmodel, f_layers = [1024,1024], nclasses = classes)
final_model.compile(adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])
final_model.summary()

final_model.fit_generator(train_generator, epochs = nepochs, steps_per_epoch = train_generator.n // bsizes, shuffle  = True)
final_model.save(sys.argv[2])
