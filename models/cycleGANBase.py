from tensorflow.keras import Model
import tensorflow as tf
import os
import cv2
from libs.misc.z_helpers_metric import * 
import tensorflow.keras as k
import time
from matplotlib import pyplot as plt
import numpy as np
import datetime 

class cycleGANBase(Model):
    def __init__ (self, **kwargs):
        Model.__init__(self, **kwargs)

    def compile(self, optimizer, lamda, learning_rate):
        self.lamda = lamda
        self.loss_object = k.losses.BinaryCrossentropy(from_logits=True)
        self.lr = learning_rate

        if optimizer=='Adam':
            #learning_rate=2e-4
            self.generator_g_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
            self.generator_r_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
            self.discriminator_a_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
            self.discriminator_b_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
        elif optimizer=='Adagrad':
            #learning_rate=0.001
            self.generator_g_optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate, initial_accumulator_value=0.1, epsilon=1e-07)
            self.generator_r_optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate, initial_accumulator_value=0.1, epsilon=1e-07)
            self.discriminator_a_optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate, initial_accumulator_value=0.1, epsilon=1e-07)
            self.discriminator_b_optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate, initial_accumulator_value=0.1, epsilon=1e-07)
        elif optimizer=='SGD': 
            #learning_rate=0.01
            self.generator_g_optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.0, nesterov=False, name='SGD')
            self.generator_r_optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.0, nesterov=False, name='SGD')
            self.discriminator_a_optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.0, nesterov=False, name='SGD')
            self.discriminator_b_optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.0, nesterov=False, name='SGD')
        elif optimizer=='RMSprop':
            #learning_rate=0.001
            self.generator_g_optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9, momentum=0.0, epsilon=1e-07)
            self.generator_r_optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9, momentum=0.0, epsilon=1e-07)
            self.discriminator_a_optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9, momentum=0.0, epsilon=1e-07)
            self.discriminator_b_optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9, momentum=0.0, epsilon=1e-07)

    def discriminator_loss(self, real, generated):
        real_loss = self.loss_object(tf.ones_like(real), real)
        generated_loss = self.loss_object(tf.zeros_like(generated), generated)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss * 0.5

    def generator_loss(self, generated):
        return self.loss_object(tf.ones_like(generated), generated)

    def calc_cycle_loss(self, real_image, cycled_image):
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
        return self.lamda * loss1

    def identity_loss(self, real_image, same_image):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return self.lamda * 0.5 * loss

    def write_jpeg(self, data, filepath):
        g = tf.Graph()
        with g.as_default():
            data_t = tf.placeholder(tf.uint8)
            op = tf.image.encode_jpeg(data_t, format='rgb', quality=100)
            init = tf.initialize_all_variables()

        with tf.Session(graph=g) as sess:
            sess.run(init)
            data_np = sess.run(op, feed_dict={ data_t: data })

        with open(filepath, 'w') as fd:
            fd.write(data_np)

    def TF2CV(self, im):
            img = tf.cast(tf.math.scalar_mul(255/2, im[0]+1), dtype=tf.uint8)
            img_ = np.array(k.utils.array_to_img(img),dtype='uint8')
            img_ = cv2.cvtColor(img_, cv2.COLOR_RGB2BGR)
            return img_

    def image_similarity(self, im1, im2, d=256):

        im1 = self.TF2CV(im1)
        im2 = self.TF2CV(im2)
        imBr_ = cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)        
        imBg_ = cv2.cvtColor(im2, cv2.COLOR_BGR2HSV)

        hBr = imBr_[:,:,0]
        hBg = imBg_[:,:,0]
        hist_2d, x_edges, y_edges = np.histogram2d(
            hBr.ravel(),
            hBg.ravel(),
            bins=d)
        mi = mutual_information(hist_2d)

        h = hBr.flatten()
        t = hBg.flatten()

        h = hBr.ravel()
        t = hBg.ravel()

        h2d_ht, _, _ = np.histogram2d(h.ravel(), t.ravel(), bins=d, normed=True)

        nmi = nmi_evaluate(h2d_ht)

        return mi, nmi

    def segmentation_accuracy(self, im1, im2):
        image_tf_1 = self.TF2CV(im1)
        image_tf_2 = self.TF2CV(im2)

        h_ = cv2.cvtColor(image_tf_1, cv2.COLOR_BGR2HSV)
        t_ = cv2.cvtColor(image_tf_2, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([89, 70, 70]).astype(np.uint8)
        upper_blue = np.array([140, 255, 255]).astype(np.uint8)
        h_mask = cv2.inRange(h_, lower_blue, upper_blue)
        t_mask = cv2.inRange(t_, lower_blue, upper_blue)
        mask1 = h_mask
        mask2 = t_mask
        intersection = cv2.bitwise_and(mask1,mask2)
        union = cv2.bitwise_or(mask1,mask2)
        notunion = cv2.bitwise_not(union)
        accuracy = (np.sum(intersection) + np.sum(notunion))/(mask1.size*255)
        _, bluemask2 = saveContouredImageFiltered(image_tf_1, h_mask, None ,64)
        _, bluemaskaug2 = saveContouredImageFiltered(image_tf_2, t_mask, None ,64)
        dice = getDice(bluemask2, bluemaskaug2)
        return accuracy, dice
    
    def evaluate_metrics(self, example_B, predict_B):
        try:
            mi, nmi = self.image_similarity(example_B, predict_B)
        except:
            mi, nmi = (-1,-1)
        try:
            acc, dsc = self.segmentation_accuracy(example_B, predict_B)
        except:
            acc, dsc = (-1,-1)

        return mi, nmi, acc, dsc


    def store_checkpoint_image(self, model, input_image, epoch, iteration, experiment_id):
        output_image = model(input_image)
        filepath = os.path.join(self.checkpoint_dir, f'{experiment_id}_{epoch}_{iteration}.jpg')
        print(f'Save_Image_Checkpoint: {iteration}')
        img = tf.cast(tf.math.scalar_mul(255/2, output_image[0]+1), dtype=tf.uint8)
        img_ = np.array(k.utils.array_to_img(img),dtype='uint8')
        img_ = cv2.cvtColor(img_, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, img_)

    def generate_images(self, model, test_input, tar):   #MAY BE REMOVED 
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

    def display_images(self, model, test_input, tar):   #MAY BE REMOVED 
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
    
    def evaluate_loss(self, val_ds):
        def evaluate_loss_sample(real_a, real_b):

            fake_b = self.generator_g(real_a, training=False)
            fake_a = self.generator_r(real_b, training=False)

            disc_real_a = self.discriminator_a(real_a, training=False)
            disc_real_b = self.discriminator_b(real_b, training=False)

            disc_fake_a = self.discriminator_a(fake_a, training=False)
            disc_fake_b = self.discriminator_b(fake_b, training=False)

            gen_g_loss = self.generator_loss(disc_fake_b)
            gen_r_loss = self.generator_loss(disc_fake_a)

            disc_a_loss = self.discriminator_loss(disc_real_a, disc_fake_a)
            disc_b_loss = self.discriminator_loss(disc_real_b, disc_fake_b)

            same_a = self.generator_r(real_a, training=False)
            same_b = self.generator_g(real_b, training=False)

            cycled_a = self.generator_r(fake_b, training=False)
            cycled_b = self.generator_g(fake_a, training=False)

            total_cycle_loss = self.calc_cycle_loss(real_a, cycled_a) + self.calc_cycle_loss(real_b, cycled_b)

            total_gen_g_loss = gen_g_loss + total_cycle_loss + self.identity_loss(real_b, same_b)
            total_gen_r_loss = gen_r_loss + total_cycle_loss + self.identity_loss(real_a, same_a)

            return gen_g_loss, gen_r_loss, disc_a_loss, disc_b_loss, total_gen_g_loss, total_gen_r_loss

        def average_loss(losses):
            return sum(losses)/len(losses)

        gen_g_loss = []
        gen_r_loss = []
        disc_a_loss = [] 
        disc_b_loss = []
        total_gen_g_loss = []
        total_gen_r_loss = []
        for _, (input_image, target) in val_ds.enumerate():
            gen_g_loss_, gen_r_loss_, disc_a_loss_, disc_b_loss_, total_gen_g_loss_, total_gen_r_loss_ = evaluate_loss_sample(input_image, target)
            gen_g_loss.append(gen_g_loss_)
            gen_r_loss.append(gen_r_loss_)
            disc_a_loss.append(disc_a_loss_)
            disc_b_loss.append(disc_b_loss_)
            total_gen_g_loss.append(total_gen_g_loss_)
            total_gen_r_loss.append(total_gen_r_loss_)
        return average_loss(gen_g_loss), average_loss(gen_r_loss), average_loss(disc_a_loss), average_loss(disc_b_loss), average_loss(total_gen_g_loss), average_loss(total_gen_r_loss)

    #@tf.function
    def train_step(self, real_a, real_b):
        # persistent is set to True because the tape is used more than
        # once to calculate the gradients.
        with tf.GradientTape(persistent=True) as tape:
            # Generator G translates A -> B
            # Generator R translates B -> A.

            fake_b = self.generator_g(real_a, training=True)
            cycled_a = self.generator_r(fake_b, training=True)

            fake_a = self.generator_r(real_b, training=True)
            cycled_b = self.generator_g(fake_a, training=True)

            # same_a and same_b are used for identity loss.
            same_a = self.generator_r(real_a, training=True)
            same_b = self.generator_g(real_b, training=True)

            disc_real_a = self.discriminator_a(real_a, training=True)
            disc_real_b = self.discriminator_b(real_b, training=True)

            disc_fake_a = self.discriminator_a(fake_a, training=True)
            disc_fake_b = self.discriminator_b(fake_b, training=True)

            # calculate the loss
            gen_g_loss = self.generator_loss(disc_fake_b)
            gen_r_loss = self.generator_loss(disc_fake_a)

            total_cycle_loss = self.calc_cycle_loss(real_a, cycled_a) + self.calc_cycle_loss(real_b, cycled_b)

            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = gen_g_loss + total_cycle_loss + self.identity_loss(real_b, same_b)
            total_gen_r_loss = gen_r_loss + total_cycle_loss + self.identity_loss(real_a, same_a)

            disc_a_loss = self.discriminator_loss(disc_real_a, disc_fake_a)
            disc_b_loss = self.discriminator_loss(disc_real_b, disc_fake_b)

        # Calculate the gradients for generator and discriminator
        generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                                self.generator_g.trainable_variables)
        generator_r_gradients = tape.gradient(total_gen_r_loss, 
                                                self.generator_r.trainable_variables)

        discriminator_a_gradients = tape.gradient(disc_a_loss, 
                                                    self.discriminator_a.trainable_variables)
        discriminator_b_gradients = tape.gradient(disc_b_loss, 
                                                    self.discriminator_b.trainable_variables)

        # Apply the gradients to the optimizer
        self.generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                                    self.generator_g.trainable_variables))

        self.generator_r_optimizer.apply_gradients(zip(generator_r_gradients, 
                                                    self.generator_r.trainable_variables))

        self.discriminator_a_optimizer.apply_gradients(zip(discriminator_a_gradients,
                                                        self.discriminator_a.trainable_variables))

        self.discriminator_b_optimizer.apply_gradients(zip(discriminator_b_gradients,
                                                        self.discriminator_b.trainable_variables))

        return gen_g_loss, gen_r_loss, disc_a_loss, disc_b_loss, total_gen_g_loss, total_gen_r_loss

    def config_checkpoints(self):

        self.log_dir=f"{self.output_dir}/log/"
        self.create_dir(self.log_dir)
        self.summary_writer = tf.summary.create_file_writer(
            self.log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.checkpoint_dir = f"{self.output_dir}/training_checkpoints/" 
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_g_optimizer=self.generator_g_optimizer,
                                        discriminator_b_optimizer=self.discriminator_b_optimizer,
                                        generator_g=self.generator_g,
                                        discriminator_b=self.discriminator_b,
                                        generator_r_optimizer=self.generator_r_optimizer,
                                        discriminator_a_optimizer=self.discriminator_a_optimizer,
                                        generator_r=self.generator_r,
                                        discriminator_a=self.discriminator_a)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, directory=self.checkpoint_dir, max_to_keep=2, checkpoint_name=self.checkpoint_prefix)
        

    def create_dir(self, mydir):
        if not os.path.exists(mydir):
            os.mkdir(mydir)
            return 1
        else:
            return -1

    def config_loggers(self):
        # Initiate Printout
        header = 'gen_g_loss, gen_r_loss, disc_a_loss, disc_b_loss, total_gen_g_loss, total_gen_r_loss'
        print('#################################')
        print('CycleGAN Training')
        print('.................................')
        print(f'.... Experiment Code: {self.experiment_id} ......')
        print('.................................')
        print(f'HEADER: {header}')
        print('.................................')
        print('.................................')
        print('.................................')

    def log_batches(self, sample, gen_g_loss, gen_r_loss, disc_a_loss, disc_b_loss, total_gen_g_loss, total_gen_r_loss):
        print(  f'Processing Batches... Batch #: {sample}, Experiment_ID: {self.experiment_id}, gen_g_loss: {gen_g_loss}, gen_r_loss: {gen_r_loss}, disc_a_loss: {disc_a_loss}, disc_b_loss:{disc_b_loss}, total_gen_g_loss: {total_gen_g_loss}, total_gen_r_loss:{total_gen_r_loss}')

    def log_epochs(self, epoch, gen_g_loss, gen_r_loss, disc_a_loss, disc_b_loss, total_gen_g_loss, total_gen_r_loss):
        print(  f'Processing Epochs... Line 1 Training Loss--> Epoch #: {epoch}, Experiment_ID: {self.experiment_id}, gen_g_loss: {gen_g_loss}, gen_r_loss: {gen_r_loss}, disc_a_loss: {disc_a_loss}, disc_b_loss:{disc_b_loss}, total_gen_g_loss: {total_gen_g_loss}, total_gen_r_loss:{total_gen_r_loss}')
        gen_g_loss_val, gen_r_loss_val, disc_a_loss_val, disc_b_loss_val, total_gen_g_loss_val, total_gen_r_loss_val = self.evaluate_loss(self.val_ds)
        print(  f'Processing Epochs... Line 2 Validation Loss--> Epoch #: {epoch}, Experiment_ID: {self.experiment_id}, gen_g_loss_val: {gen_g_loss_val}, gen_r_loss_val: {gen_r_loss_val}, disc_a_loss_val: {disc_a_loss_val}, disc_b_loss_val:{disc_b_loss_val}, total_gen_g_loss_val: {total_gen_g_loss_val}, total_gen_r_loss_val:{total_gen_r_loss_val}')

    def writeMonitorImage(self, imgs, fname):
        input_image, target, gen_output = imgs

        def format_output(imgs):
            input_image, target, gen_output = imgs
            out = cv2.hconcat((input_image,target))
            out = cv2.hconcat((out, gen_output))
            return out

        input_image = self.TF2CV(input_image)
        target = self.TF2CV(target)
        gen_output = self.TF2CV(gen_output)

        out = format_output((input_image, target, gen_output ))

        cv2.imwrite(f'{self.monitor_dir}/{self.experiment_id}_{fname}.jpg', out)

    def config_monitor(self):
        self.monitor_dir=f"{self.output_dir}/monitor_output/"
        self.create_dir(self.monitor_dir)


    def process_monitor(self, epoch, sample):

        def get_prediction(model, test_input, tar):   #MAY BE REMOVED 
            prediction = model(test_input, training=True)
            return prediction

        for i, (input_image, target) in self.monitor_ds.enumerate():
            #self.display_images(self.generator, input_image, target)
            gen_output = get_prediction(self.generator_g, input_image, target)
            self.writeMonitorImage((input_image, target, gen_output), f'{epoch}_{sample}_{i}')
        
    def config_modelSave(self):
        self.modelSave=f"{self.output_dir}/saved_Models/"
        self.create_dir(self.modelSave)
    
    def saveModel(self, epoch):
        k.models.save_model(self.generator_g, os.path.join(self.modelSave, f'{self.experiment_id}_{epoch}.h5'), save_format='h5')

    def fit(self, train_ds, val_ds, monitor_ds, epochs, experiment_id, dataroot, monitor_freq='No Monitoring', checkpointfreq=5, modelsavefreq = 5):

        trainA, trainB = train_ds

        self.output_dir = f'{dataroot}/output_{experiment_id}'

        self.create_dir(self.output_dir)

        self.val_ds = val_ds
        self.monitor_ds = monitor_ds
        self.experiment_id = experiment_id
        self.dataroot = dataroot

        self.config_monitor()
        self.config_checkpoints()
        self.config_loggers()
        self.config_modelSave()

        batches = len(trainA)
        print_freq = batches//10

        # Iterate Epochs
        for epoch in range(epochs):      
            start = time.time()
            print(f'Epoch Started .... Time: {start}, Epoch # {epoch}')
            n = 0
            # Iterate Batches
            for image_x, image_y in tf.data.Dataset.zip((trainA, trainB)):
                gen_g_loss, gen_r_loss, disc_a_loss, disc_b_loss, total_gen_g_loss, total_gen_r_loss = self.train_step(image_x, image_y)

                if n%print_freq==0:
                    self.log_batches(n, gen_g_loss, gen_r_loss, disc_a_loss, disc_b_loss, total_gen_g_loss, total_gen_r_loss)
                    if monitor_freq=='batch':
                        self.process_monitor(epoch, n)
                n += 1
            
            # End of Epoch Reporting and Logging
            self.log_epochs(epoch, gen_g_loss, gen_r_loss, disc_a_loss, disc_b_loss, total_gen_g_loss, total_gen_r_loss)

            if monitor_freq=='epoch' or monitor_freq=='batch':
                self.process_monitor(epoch, -1)
            else:
                try:
                    freq = int(monitor_freq.replace('epoch', ''))
                    if epoch%freq==0:
                        self.process_monitor(epoch, -1)
                except:
                    pass

            try:
                if epoch%checkpointfreq==0:
                    self.checkpoint_manager.save()
            except:
                print('Epoch CheckPoint Error')

            try:
                if epoch%modelsavefreq==0 and epoch>0:
                    self.saveModel(epoch)
            except:
                print('Epoch Save Model Error')

            #print(f'Processing Epochs... Error in Epoch #: {epoch}')
        self.saveModel('fully_trained')

    def log_testing(self, sample, testing_metric):
        mi, nmi, acc, dice = testing_metric
        print(  f'Processing Testing... Image #: {sample}, Experiment_ID: {self.experiment_id}, mi: {mi}, nmi: {nmi}, acc: {acc}, dice:{dice}')
    
    def saveTestOutput(self, imgs, fname):
        gen_img = self.TF2CV(imgs)
        cv2.imwrite(f'{self.out_img_dir}/{fname}.jpg', gen_img)

    def processTestSample(self, input_image, target, image_file):
        def get_prediction(model, test_input, tar):   #MAY BE REMOVED 
            prediction = model(test_input, training=True)
            return prediction
        gen_output = get_prediction(self.generator_g, input_image, target)
        fname = str(image_file.numpy()[0]).replace("'", "").split('\\')[-1].replace('.jpg', '')
        self.saveTestOutput(gen_output, fname)
        testing_metrics = self.evaluate_metrics(target, gen_output)
        return testing_metrics

    def test(self, test_dataset, experiment_id, dataroot):
        self.experiment_id = experiment_id
        self.dataroot = dataroot
        self.output_dir = f'{dataroot}/output_{experiment_id}'
        self.create_dir(self.output_dir)
        self.out_img_dir = f'{self.output_dir}/predictions'
        self.create_dir(self.out_img_dir)

        # Iterate Batches
        for sample, (input_image, target, image_file) in test_dataset.enumerate():
            testing_metric = self.processTestSample(input_image, target, image_file)
            self.log_testing(sample, testing_metric)