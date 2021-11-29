from tensorflow.keras import backend as k
from tensorflow.keras.applications import vgg19
import utils as u


def load_model(target_image_path, style_reference_image_path, img_height, img_width):
    target_image = k.constant(u.preprocess_image(target_image_path, img_height, img_width))
    style_reference_image = k.constant(u.preprocess_image(style_reference_image_path, img_height, img_width))
    combination_image = k.placeholder((1, img_height, img_width, 3))

    input_tensor = k.concatenate([target_image, style_reference_image, combination_image], axis=0)

    # exclude dense layer
    model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)

    print('Model loaded.')

    return model, combination_image


# the l2 of the activation of generated image and target image on the top convolution layer
def content_loss(base, combination):
    return k.sum(k.square(combination - base))


# dot product of features
def gram_matrix(x):
    features = k.batch_flatten(k.permute_dimensions(x, (2, 0, 1)))
    gram = k.dot(features, k.transpose(features))
    return gram


# Gram matrix measures the relevance of different features
def style_loss(style, combination, img_height, img_width):
    s = gram_matrix(style)
    c = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return k.sum(k.square(s - c)) / (4. * (channels ** 2) * (size ** 2))


def total_variation_loss(x, img_height, img_width):
    a = k.square(x[:, :img_height - 1, :img_width - 1, :] - x[:, 1:, :img_width - 1, :])
    b = k.square(x[:, :img_height - 1, :img_width - 1, :] - x[:, :img_height - 1, 1:, :])
    return k.sum(k.pow(a + b, 1.25))
