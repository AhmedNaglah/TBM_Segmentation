import tensorflow as tf
from libs.generative.utils import *


def unet_generator_64(channels = 3, norm_type='batchnorm'):
    inputs = tf.keras.layers.Input(shape=[64, 64, channels])

    down_stack = [
        downsample(256, 4, norm_type, apply_norm=False),  # (batch_size, 32, 32, 256)
        downsample(512, 4, norm_type),  # (batch_size, 16, 16, 512)
        downsample(512, 4, norm_type),  # (batch_size, 8, 8, 512)
        downsample(512, 4, norm_type),  # (batch_size, 4, 4, 512)
        downsample(512, 4, norm_type),  # (batch_size, 2, 2, 512)
        downsample(512, 4, norm_type),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, norm_type, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        upsample(512, 4, norm_type, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample(512, 4, norm_type, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample(512, 4, norm_type),  # (batch_size, 16, 16, 1024)
        upsample(256, 4, norm_type),  # (batch_size, 32, 32, 512)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(channels, 4,
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            activation='tanh')  # (batch_size, 64, 64, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def unet_generator_128(channels = 3, norm_type='batchnorm'):
    inputs = tf.keras.layers.Input(shape=[128, 128, channels])

    down_stack = [
        downsample(128, 4, norm_type, apply_norm=False),  # (batch_size, 64, 64, 128)
        downsample(256, 4, norm_type),  # (batch_size, 32, 32, 256)
        downsample(512, 4, norm_type),  # (batch_size, 16, 16, 512)
        downsample(512, 4, norm_type),  # (batch_size, 8, 8, 512)
        downsample(512, 4, norm_type),  # (batch_size, 4, 4, 512)
        downsample(512, 4, norm_type),  # (batch_size, 2, 2, 512)
        downsample(512, 4, norm_type),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, norm_type, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        upsample(512, 4, norm_type, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample(512, 4, norm_type, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample(512, 4, norm_type),  # (batch_size, 16, 16, 1024)
        upsample(256, 4, norm_type),  # (batch_size, 32, 32, 512)
        upsample(128, 4, norm_type),  # (batch_size, 64, 64, 256)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(channels, 4,
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            activation='tanh')  # (batch_size, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def unet_generator_256(channels = 3, norm_type='batchnorm'):
    inputs = tf.keras.layers.Input(shape=[256, 256, channels])

    down_stack = [
        downsample(64, 4, norm_type, apply_norm=False),  # (batch_size, 128, 128, 64)
        downsample(128, 4, norm_type),  # (batch_size, 64, 64, 128)
        downsample(256, 4, norm_type),  # (batch_size, 32, 32, 256)
        downsample(512, 4, norm_type),  # (batch_size, 16, 16, 512)
        downsample(512, 4, norm_type),  # (batch_size, 8, 8, 512)
        downsample(512, 4, norm_type),  # (batch_size, 4, 4, 512)
        downsample(512, 4, norm_type),  # (batch_size, 2, 2, 512)
        downsample(512, 4, norm_type),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, norm_type, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        upsample(512, 4, norm_type, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample(512, 4, norm_type, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample(512, 4, norm_type),  # (batch_size, 16, 16, 1024)
        upsample(256, 4, norm_type),  # (batch_size, 32, 32, 512)
        upsample(128, 4, norm_type),  # (batch_size, 64, 64, 256)
        upsample(64, 4, norm_type),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(channels, 4,
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            activation='tanh')  # (batch_size, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def unet_generator_512(channels = 3, norm_type='batchnorm'):
    inputs = tf.keras.layers.Input(shape=[512, 512, channels])

    down_stack = [
        downsample(64, 4, norm_type, apply_norm=False),  # (batch_size, 256, 256, 64)
        downsample(64, 4, norm_type),  # (batch_size, 128, 128, 64)
        downsample(128, 4, norm_type),  # (batch_size, 64, 64, 128)
        downsample(256, 4, norm_type),  # (batch_size, 32, 32, 256)
        downsample(512, 4, norm_type),  # (batch_size, 16, 16, 512)
        downsample(512, 4, norm_type),  # (batch_size, 8, 8, 512)
        downsample(512, 4, norm_type),  # (batch_size, 4, 4, 512)
        downsample(512, 4, norm_type),  # (batch_size, 2, 2, 512)
        downsample(512, 4, norm_type),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, norm_type, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        upsample(512, 4, norm_type, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample(512, 4, norm_type, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample(512, 4, norm_type),  # (batch_size, 16, 16, 1024)
        upsample(256, 4, norm_type),  # (batch_size, 32, 32, 512)
        upsample(128, 4, norm_type),  # (batch_size, 64, 64, 256)
        upsample(64, 4, norm_type),  # (batch_size, 128, 128, 128)
        upsample(64, 4, norm_type),  # (batch_size, 256, 256, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(channels, 4,
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            activation='tanh')  # (batch_size, 512, 512, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def unet_generator_1024(channels = 3, norm_type='batchnorm'):
    inputs = tf.keras.layers.Input(shape=[1024, 1024, channels])

    down_stack = [
        downsample(64, 4, norm_type, apply_norm=False),  # (batch_size, 512, 512, 64)
        downsample(64, 4, norm_type),  # (batch_size, 256, 256, 64)
        downsample(64, 4, norm_type),  # (batch_size, 128, 128, 64)
        downsample(128, 4, norm_type),  # (batch_size, 64, 64, 128)
        downsample(128, 4, norm_type),  # (batch_size, 32, 32, 256)
        downsample(256, 4, norm_type),  # (batch_size, 16, 16, 512)
        downsample(512, 4, norm_type),  # (batch_size, 8, 8, 512)
        downsample(512, 4, norm_type),  # (batch_size, 4, 4, 512)
        downsample(512, 4, norm_type),  # (batch_size, 2, 2, 512)
        downsample(512, 4, norm_type),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, norm_type, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        upsample(512, 4, norm_type, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample(512, 4, norm_type, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample(256, 4, norm_type),  # (batch_size, 16, 16, 512)
        upsample(128, 4, norm_type),  # (batch_size, 32, 32, 512)
        upsample(128, 4, norm_type),  # (batch_size, 64, 64, 256)
        upsample(64, 4, norm_type),  # (batch_size, 128, 128, 128)
        upsample(64, 4, norm_type),  # (batch_size, 256, 256, 128)
        upsample(64, 4, norm_type),  # (batch_size, 512, 512, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(channels, 4,
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            activation='tanh')  # (batch_size, 1024, 1024, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
