''' Kishan Patel '''

import os
import sys
import keras
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

import keras.backend as K
K.set_image_data_format('channels_last')
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"


test_path = sys.argv[1]
mmodel = load_model(sys.argv[2])

test_data_generator = ImageDataGenerator(preprocessing_function = preprocess_input)
test_generator = test_data_generator.flow_from_directory(test_path, target_size = (256, 256), batch_size = 8)

# Model Summary
print(mmodel.summary())

# Evaluating the Model with Test Set
preds = mmodel.evaluate_generator(test_generator)
error = (1 - preds[1])*100
acc = (preds[1])*100

print("Test Error: \t\t{}%".format(error))
print("Test Accuracy: \t\t{}%".format(acc))

f = open("Chest_Xray_Output.txt", "w")
terror = (1 - preds[1]) * 100
taccuracy = (preds[1]) * 100
s1 = "Test Error: \t\t" + str(terror) + "%"
s2 = "Test Accuracy: \t\t" + str(taccuracy) + "%"
f.write("\nModel Summary\n")
mmodel.summary(print_fn=lambda x: f.write(x + '\n'))
f.write(s1)
f.write('\n')
f.write(s2)
f.close()

