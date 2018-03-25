import os
import cv2
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras import models
import numpy as np
import matplotlib.pyplot as plt

def visualize_inter_activations(img_path, model, no_layers):
    # Extracts the outputs of the top no_layers layers:
    img = cv2.imread(img_path)
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    # Remember that the model was trained on inputs that were preprocessed in the following way:
    img_tensor /= 255.
    print(img_tensor.shape)

    # Extracts the outputs of the top no_layers layers:
    layer_outputs = [layer.output for layer in model.layers[:no_layers]]
    # Creates a model that will return these outputs, given the model input:
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    # This will return a list of several Numpy arrays:
    # one array per layer activation
    activations = activation_model.predict(img_tensor)

    # These are the names of the layers, so can have them as part of our plot
    layer_names = []
    for layer in model.layers[:no_layers]:
        layer_names.append(layer.name)
        images_per_row = 8
    # Now let's display our feature maps
    for layer_name, layer_activation in zip(layer_names, activations):
        # This is the number of features in the feature map
        n_features = layer_activation.shape[-1]
        # The feature map has shape (1, size, size, n_features)
        size = layer_activation.shape[1]
        # We will tile the activation channels in this matrix
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        # We'll tile each filter into this big horizontal grid
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                :, :,
                col * images_per_row + row]
                # Post-process the feature to make it visually palatable
                channel_image -= channel_image.mean()
                channel_image /= (channel_image.std()+0.000000001)
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size,
                row * size : (row + 1) * size] = channel_image
        # Display the grid
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
    
    fig = plt.gcf()
    plt.show()
    fig.savefig('test_visualize_inter_activations.png')
