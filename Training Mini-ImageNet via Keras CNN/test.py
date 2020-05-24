import sys; import keras; import numpy as np
from keras.utils import to_categorical
from keras.models import load_model
import keras.backend as K
K.set_image_data_format('channels_first')

# Load Test Images and Labels
X_test = np.load(sys.argv[1])
Y_test = np.load(sys.argv[2])

# Normlaize Test Images
normX_test = X_test/255.

# Encode the Labels
catY_test = to_categorical(Y_test, 10)

# Load the Model
mmodel = load_model(sys.argv[3])
print("\nModel Summary: \n")
print(mmodel.summary())

# Evaluate the Test Set
predictions = mmodel.evaluate(x = normX_test, y = catY_test, sample_weight = None)
print("Test Error: \t\t{} %".format((1 - predictions[1])*100))
print("Test Accuracy: \t\t{} %".format((predictions[1])*100))

f = open("Output.txt", "w")
terror = (1 - predictions[1]) * 100
taccuracy = (predictions[1]) * 100
s1 = "Test Error: \t\t" + str(terror) + "%"
s2 = "Test Accuracy: \t\t" + str(taccuracy) + "%"
f.write("\nModel Summary\n")
mmodel.summary(print_fn=lambda x: f.write(x + '\n'))
f.write(s1)
f.write('\n')
f.write(s2)
f.close()
