import tensorflow as tf
import cv2
import numpy as np

def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image

def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image

def load_paired(image_file):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image)

  w = tf.shape(image)[1]
  w = w // 2
  input_image = image[:, :w, :]
  real_image = image[:, w:, :]

  input_image = tf.cast(input_image, tf.float32)
  real_image = tf.cast(real_image, tf.float32)

  return input_image, real_image

def load_triple(image_file):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image)

  w = tf.shape(image)[1]
  w = w // 3
  input_image = image[:, :w, :]
  gt = image[:, w:2*w, :]
  prediction = image[:, 2*w:, :]

  input_image = tf.cast(input_image, tf.float32)
  gt = tf.cast(gt, tf.float32)
  prediction = tf.cast(prediction, tf.float32)

  return input_image, gt, prediction

def load_image(image_file):
  input_image, real_image = load_paired(image_file)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

def load_image_segmentation(image_file):
  input_image, real_image = load_triple(image_file)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

def load_image_test(image_file):
  input_image, real_image = load_paired(image_file)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image, image_file

def load_png(image_file):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image)

  w = tf.shape(image)[1]
  w = w // 2
  input_image = image[:, :w, :]
  real_image = image[:, w:, :]

  input_image = tf.cast(input_image, tf.float32)
  real_image = tf.cast(real_image, tf.float32)

  return input_image, real_image

def load_image_png(image_file):
  input_image, real_image = load_png(image_file)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

def load_png_segmentation(image_file):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image)

  w = tf.shape(image)[1]
  w = w // 2
  input_image = image[:, :w, :]
  real_image = image[:, w:, :]

  input_image = tf.cast(input_image, tf.float32)
  real_image = tf.cast(real_image, tf.float32)
  
  real_image_ = tf.subtract(tf.cast(1, tf.float32), real_image)

  y = tf.stack([real_image, real_image_], axis=0)

  return input_image, y

def load_image_png_segmentation(image_file):
  input_image, real_image = load_png_segmentation(image_file)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

def normalizeCycleGAN(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image

def loadCycleGAN(image_file):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_png(image, channels=3)
  image = normalizeCycleGAN(image)
  return image

