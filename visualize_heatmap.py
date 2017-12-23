import os
import cv2
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras import models
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K

def visualize_heatmap(img_path, model, layer_name):
    img = cv2.imread(img_path)
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    # Remember that the model was trained on inputs that were preprocessed in the following way:
    img_tensor /= 255.
    print(img_tensor.shape)

    preds = model.predict(img_tensor)
    # This is the "african elephant" entry in the prediction vector
    african_elephant_output = model.output[:, np.argmax(preds[0])]
    # The is the output feature map of the `layer_name` layer,
    last_conv_layer = model.get_layer(layer_name)
    # This is the gradient of the "african elephant" class with regard to
    # the output feature map of `layer_name`
    grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]
    # This is a vector of shape (512,), where each entry
    # is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    # This function allows us to access the values of the quantities we just defined:
    # `pooled_grads` and the output feature map of `block5_conv3`,
    # given a sample image
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    # These are the values of these two quantities, as Numpy arrays,
    # given our sample image of two elephants
    pooled_grads_value, conv_layer_output_value = iterate([img_tensor])
    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the elephant class
    for i in range(256):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(conv_layer_output_value, axis=-1)

    # We use cv2 to load the original image
    # img = cv2.imread(img_path)
    # We resize the heatmap to have the same size as the original image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # We convert the heatmap to RGB
    heatmap -= np.min(heatmap)
    heatmap /= np.max(heatmap)
    heatmap = np.uint8(255 * heatmap)

    # We apply the heatmap to the original image
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 0.4 here is a heatmap intensity factor
    superimposed_img = heatmap * 0.4 + img
    # Save the image to disk
    cv2.imwrite('test_heatmap.jpg', heatmap)
    cv2.imwrite('test_overlap_heatmap.jpg', superimposed_img)
