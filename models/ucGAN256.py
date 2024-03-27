from models.base import Base
import tensorflow as tf
import time
import datetime
import os
from matplotlib import pyplot as plt
import cv2
import tensorflow.keras as k
import numpy as np

OUTPUT_CHANNELS = 3
LAMBDA = 100

class ucGAN256(Base):

    def __init__(self, experiment_id='undefined_experiment', checkpntpath = './checkpoint_image_path/',**kwargs):
        Base.__init__(self, **kwargs)
        self.generator = self.Generator()
        self.discriminator = self.Discriminator()
        self.checkpoint_image_path = checkpntpath
        self.experiment_id = experiment_id
        if not os.path.exists(self.checkpoint_image_path):
            os.mkdir(self.checkpoint_image_path)

    def call(self, inputs):
        x2_ = self.Generator(inputs)
        return x2_

    def compile(self, optimizer='Adam', lamda=100, learning_rate=2e-4):
        self.lamda = lamda

        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        if optimizer=='Adam':
            #learning_rate=2e-4
            self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
            self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
        elif optimizer=='Adagrad':
            #learning_rate=0.001
            self.generator_optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate, initial_accumulator_value=0.1, epsilon=1e-07)
            self.discriminator_optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate, initial_accumulator_value=0.1, epsilon=1e-07)
        elif optimizer=='SGD': 
            #learning_rate=0.01
            self.generator_optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.0, nesterov=False, name='SGD')
            self.discriminator_optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.0, nesterov=False, name='SGD')
        elif optimizer=='RMSprop':
            #learning_rate=0.001
            self.generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9, momentum=0.0, epsilon=1e-07)
            self.discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9, momentum=0.0, epsilon=1e-07)

        self.log_dir="logs/"
        self.summary_writer = tf.summary.create_file_writer(
            self.log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                        discriminator_optimizer=self.discriminator_optimizer,
                                        generator=self.generator,
                                        discriminator=self.discriminator)
        return         

    def downsample(self, filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                    kernel_initializer=initializer, use_bias=False))

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())

        return result

    def upsample(self, filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))

        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())

        return result

    def Generator(self):
        inputs = tf.keras.layers.Input(shape=[256, 256, 3])

        down_stack = [
            self.downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
            self.downsample(128, 4),  # (batch_size, 64, 64, 128)
            self.downsample(256, 4),  # (batch_size, 32, 32, 256)
            self.downsample(512, 4),  # (batch_size, 16, 16, 512)
            self.downsample(512, 4),  # (batch_size, 8, 8, 512)
            self.downsample(512, 4),  # (batch_size, 4, 4, 512)
            self.downsample(512, 4),  # (batch_size, 2, 2, 512)
            self.downsample(512, 4),  # (batch_size, 1, 1, 512)
        ]

        up_stack = [
            self.upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
            self.upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
            self.upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
            self.upsample(512, 4),  # (batch_size, 16, 16, 1024)
            self.upsample(256, 4),  # (batch_size, 32, 32, 512)
            self.upsample(128, 4),  # (batch_size, 64, 64, 256)
            self.upsample(64, 4),  # (batch_size, 128, 128, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
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

    def generator_loss(self, disc_generated_output, gen_output, target):

        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

        # Mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        total_gen_loss = gan_loss + (LAMBDA * l1_loss)

        return total_gen_loss, gan_loss, l1_loss

    def Discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)

        tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

        down1 = self.downsample(64, 4, False)(tar)  # (batch_size, 128, 128, 64)
        down2 = self.downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
        down3 = self.downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                        kernel_initializer=initializer,
                                        use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

        last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                        kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

        return tf.keras.Model(inputs=tar, outputs=last)

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)

        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss

    def store_checkpoint_image(self, model, input_image, epoch, iteration, experiment_id):
        output_image = model(input_image)
        filepath = os.path.join(self.checkpoint_image_path, f'{experiment_id}_{epoch}_{iteration}.jpg')
        print(f'Save_Image_Checkpoint: {iteration}')
        img = tf.cast(tf.math.scalar_mul(255/2, output_image[0]+1), dtype=tf.uint8)
        img_ = np.array(k.utils.array_to_img(img),dtype='uint8')
        img_ = cv2.cvtColor(img_, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, img_)

    def generate_images(self, model, test_input, tar):
        prediction = model(test_input, training=True)
        plt.figure(figsize=(15, 15))

        display_list = [test_input[0], tar[0], prediction[0]]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']

        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.title(title[i])
            # Getting the pixel values in the [0, 1] range to plot.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        plt.show()

    def display_images(self, model, test_input, tar):
        prediction = model(test_input, training=True)
        plt.figure(figsize=(15, 15))

        display_list = [test_input[0], tar[0], prediction[0]]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']

        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.title(title[i])
            # Getting the pixel values in the [0, 1] range to plot.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        plt.show()

    @tf.function
    def train_step(self, input_image, target):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator(target, training=True)
            disc_generated_output = self.discriminator(gen_output, training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(disc_generated_output, gen_output, target)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                    self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                    self.discriminator.trainable_variables))

        return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss

    def fit(self, train_ds, test_ds, epochs=100):
        example_input, example_target = next(iter(test_ds.take(1)))

        # Initiate Logger
        header = 'gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss'
        loggername = 'condGAN_hyper_fit_epochs'
        logfile = os.path.join(self.checkpoint_image_path, f'{loggername}.txt')
        self.initiate_logger(loggername, header, logfile)

        for epoch in range(epochs):        
            start = time.time()
            for sample, (input_image, target) in train_ds.enumerate():
                gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = self.train_step(input_image, target)
                if sample%100==0:
                    #self.write_log(loggername, f'{gen_total_loss} , {gen_gan_loss} , {gen_l1_loss} , {disc_loss}')
                    print(f'Sample: {sample}')
            
            self.store_checkpoint_image(self.generator, example_input, epoch, sample, self.experiment_id)
            self.write_log(loggername, f'{gen_total_loss} , {gen_gan_loss} , {gen_l1_loss} , {disc_loss}')
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)
            print(f'Epoch: {epoch};  Time taken for 1 epoch: {time.time()-start:.2f} sec;  gen_total_loss: {gen_total_loss};  gen_gan_loss: {gen_gan_loss};  gen_l1_loss: {gen_l1_loss};  disc_loss: {disc_loss}\n')
            #self.generate_images(self.generator, example_input, example_target)
            if epoch%10==0:
                k.models.save_model(self.generator, os.path.join(self.checkpoint_image_path, f'{self.experiment_id}_{epoch}.h5'), save_format='h5')
            epoch_time = time.time() - start
            print(f'epoch : {epoch},  time : {epoch_time}')
