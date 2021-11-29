import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import backend as k
from scipy.optimize import fmin_l_bfgs_b
import cv2 as cv
import time
import model as m
import utils as u
import evaluator as e

tf.compat.v1.disable_eager_execution()

# image path
target_image_path = 'D:/Study/neural_style_transfer/img/target.png'  # img you want to change
style_reference_image_path = 'D:/Study/neural_style_transfer/img/style.png'  # the style you want to transfer

# original image size
width, height = load_img(target_image_path).size

# define the generated image's size
img_height = 300
img_width = int(width * img_height / height)

model, combination_image = m.load_model(target_image_path, style_reference_image_path, img_height, img_width)

outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

content_layer = 'block5_conv2'
style_layers = ['block1_conv1', 'b:lock2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

# weight
total_variation_weight = 1e-4
style_weight = 1.
content_weight = 0.025

loss = k.variable(0.)
layer_features = outputs_dict[content_layer]
target_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss = loss + content_weight * m.content_loss(target_image_features, combination_features)

# style loss
for layer_name in style_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = m.style_loss(style_reference_features, combination_features, img_height, img_width)
    loss = loss + (style_weight / len(style_layers)) * sl
    loss = loss + total_variation_weight * m.total_variation_loss(combination_image, img_height, img_width)

# gradient of loss relative to image
grads = k.gradients(loss, combination_image)
fetch_loss_and_grads = k.function([combination_image], [loss, grads])

evaluator = e.Evaluator(img_height, img_width, fetch_loss_and_grads)

result_prefix = 'result'
iterations = 20

x = u.preprocess_image(target_image_path, img_height, img_width)
x = x.flatten()
for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x, fprime=evaluator.grads, maxfun=20)

    print('Current loss value:', min_val)
    img = x.copy().reshape((img_height, img_width, 3))
    img = u.deprocess_image(img)
    fname = 'D:/Study/neural_style_transfer/img/results/' + result_prefix + '_at_iteration_%d.png' % i
    cv.imwrite(fname, img)
    print('Image saved as', fname)
    end_time = time.time()
    print('Iteration %d completed in %ds' % (i, end_time - start_time))
