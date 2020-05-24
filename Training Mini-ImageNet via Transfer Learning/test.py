
# Import Libraries
import sys; import numpy as np; import keras;
from keras.models import load_model; from keras.utils import layer_utils;
from keras.initializers import glorot_uniform; from keras.applications.vgg16 import VGG16, preprocess_input;
from keras.preprocessing.image import ImageDataGenerator;
import keras.backend as K;

# Set Format
K.set_image_data_format('channels_last')

# Inputs
test_dir = sys.argv[1]
mModel = load_model(sys.argv[2])

# Process and Generate Train Data
datagen_test = ImageDataGenerator(preprocessing_function = preprocess_input)
generate_test = datagen_test.flow_from_directory(test_dir, target_size = (256, 256), batch_size = 8)

# Print the Model Summary
print(mModel.summary())

# Evaluate
predictions = mModel.evaluate_generator(generate_test)

print("\n Test Error: {}%".format((1 - predictions[1])*100))
print("\n Test Accuracy: {}%".format((predictions[1])*100))

f = open("Output.txt", "w")
terror = (1 - predictions[1]) * 100
taccuracy = (predictions[1]) * 100
s1 = "Test Error: \t\t" + str(terror) + "%"
s2 = "Test Accuracy: \t\t" + str(taccuracy) + "%"
f.write("\nModel Summary\n")
mModel.summary(print_fn=lambda x: f.write(x + '\n'))
f.write(s1)
f.write('\n')
f.write(s2)
f.close()
