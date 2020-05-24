# Name: Kishan Patel

import keras
import os
import sys
import numpy as np
from keras.models import load_model
from numpy.random import randn
from matplotlib import pyplot

model = load_model(sys.argv[1])
save_path = 'generated_images/'
#save_path = sys.argv[2]

if not os.path.exists(save_path):
    os.makedirs(save_path)

# Generating Points in Latent Space to Generate Images With
def latent_points(latent_dim, nsamples):
    xinput = randn(latent_dim * nsamples) # Generating Samples in Latent Space
    xinput = xinput.reshape(nsamples, latent_dim) # Reshape
    return xinput

# Create and Save Images
def plot_images(examples, j, n = 10):
    for i in range(n * n):
        pyplot.subplot(n, n, 1 + i)    # define subplot
        pyplot.axis('off') # turn off axis
        pyplot.imshow(examples[i, :, :, 0], cmap='gray_r') # plot raw pixel data
    filename = '500epc_generated_plot_e%03d.png' % (j)
    pyplot.savefig(f'{save_path}test_output/{filename}')
    pyplot.close()

for i in range(50):
    lp = latent_points(100, 25)
    X = model.predict(lp)
    plot_images(X,i,5)

print("Done--Check the generated_images/test_output directory")
