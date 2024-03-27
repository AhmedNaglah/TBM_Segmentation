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

class condGANBase(Model):
    def __init__ (self, **kwargs):
        Model.__init__(self, **kwargs)

    def compile(self, optimizer, lamda, learning_rate):
        self.lamda = lamda
        self.loss_object = k.losses.BinaryCrossentropy(from_logits=True)
        self.lr = learning_rate

        if optimizer=='Adam':
            #learning_rate=2e-4
            self.generator_optimizer = k.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
            self.discriminator_optimizer = k.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
        elif optimizer=='Adagrad':
            #learning_rate=0.001
            self.generator_optimizer = k.optimizers.Adagrad(learning_rate=learning_rate, initial_accumulator_value=0.1, epsilon=1e-07)
            self.discriminator_optimizer = k.optimizers.Adagrad(learning_rate=learning_rate, initial_accumulator_value=0.1, epsilon=1e-07)
        elif optimizer=='SGD': 
            #learning_rate=0.01
            self.generator_optimizer = k.optimizers.SGD(learning_rate=learning_rate, momentum=0.0, nesterov=False, name='SGD')
            self.discriminator_optimizer = k.optimizers.SGD(learning_rate=learning_rate, momentum=0.0, nesterov=False, name='SGD')
        elif optimizer=='RMSprop':
            #learning_rate=0.001
            self.generator_optimizer = k.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9, momentum=0.0, epsilon=1e-07)
            self.discriminator_optimizer = k.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9, momentum=0.0, epsilon=1e-07)
        return      

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
    
    def evaluate_loss(self, val_ds):        #MAY BE REMOVED 
        def evaluate_loss_sample(inp, tar):
            gen_output = self.generator(inp, training=True)
            disc_real_output = self.discriminator([input_image, target], training=False)
            disc_generated_output = self.discriminator([input_image, gen_output], training=False)
            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(disc_generated_output, gen_output, tar)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)
            return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss

        def average_loss(losses):
            return sum(losses)/len(losses)

        gen_total_loss = []
        gen_gan_loss =  []
        gen_l1_loss = [] 
        disc_loss = []
        for _, (input_image, target) in val_ds.enumerate():
            gen_total_loss_, gen_gan_loss_, gen_l1_loss_, disc_loss_ = evaluate_loss_sample(input_image, target)
            gen_total_loss.append(gen_total_loss_)
            gen_gan_loss.append(gen_gan_loss_)
            gen_l1_loss.append(gen_l1_loss_)
            disc_loss.append(disc_loss_)
        return average_loss(gen_total_loss), average_loss(gen_gan_loss), average_loss(gen_l1_loss), average_loss(disc_loss)

    @tf.function
    def train_step(self, input_image, target):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)

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

    def restore_from_checkpoint(self, mycheck_point_path):
        optimizer = 'Adam'
        learning_rate = 2e-4
        if optimizer=='Adam':
            #learning_rate=2e-4
            self.generator_optimizer = k.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
            self.discriminator_optimizer = k.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
        elif optimizer=='Adagrad':
            #learning_rate=0.001
            self.generator_optimizer = k.optimizers.Adagrad(learning_rate=learning_rate, initial_accumulator_value=0.1, epsilon=1e-07)
            self.discriminator_optimizer = k.optimizers.Adagrad(learning_rate=learning_rate, initial_accumulator_value=0.1, epsilon=1e-07)
        elif optimizer=='SGD': 
            #learning_rate=0.01
            self.generator_optimizer = k.optimizers.SGD(learning_rate=learning_rate, momentum=0.0, nesterov=False, name='SGD')
            self.discriminator_optimizer = k.optimizers.SGD(learning_rate=learning_rate, momentum=0.0, nesterov=False, name='SGD')
        elif optimizer=='RMSprop':
            #learning_rate=0.001
            self.generator_optimizer = k.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9, momentum=0.0, epsilon=1e-07)
            self.discriminator_optimizer = k.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9, momentum=0.0, epsilon=1e-07)
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                discriminator_optimizer=self.discriminator_optimizer,
                                generator=self.generator,
                                discriminator=self.discriminator)
        self.checkpoint.restore(mycheck_point_path)

    def config_checkpoints(self):

        self.log_dir=f"{self.output_dir}/log/"
        self.create_dir(self.log_dir)
        self.summary_writer = tf.summary.create_file_writer(
            self.log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.checkpoint_dir = f"{self.output_dir}/training_checkpoints/" 
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                        discriminator_optimizer=self.discriminator_optimizer,
                                        generator=self.generator,
                                        discriminator=self.discriminator)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, directory=self.checkpoint_dir, max_to_keep=2, checkpoint_name=self.checkpoint_prefix)
        return

    def create_dir(self, mydir):
        if not os.path.exists(mydir):
            os.mkdir(mydir)
            return 1
        else:
            return -1

    def config_loggers(self):
        # Initiate Printout
        header = 'gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss'
        print('#################################')
        print('CondGAN Training')
        print('.................................')
        print(f'.... Experiment Code: {self.experiment_id} ......')
        print('.................................')
        print(f'HEADER: {header}')
        print('.................................')
        print('.................................')
        print('.................................')

    def log_batches(self, sample, gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss):
        print(  f'Processing Batches... Batch #: {sample}, Experiment_ID: {self.experiment_id}, gen_total_loss: {gen_total_loss}, gen_gan_loss: {gen_gan_loss}, gen_l1_loss: {gen_l1_loss}, disc_loss:{disc_loss}')

    def log_epochs(self, epoch, gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss):
        print(  f'Processing Epochs... Line 1 Training Loss--> Epoch #: {epoch}, Experiment_ID: {self.experiment_id}, gen_total_loss: {gen_total_loss}, gen_gan_loss: {gen_gan_loss}, gen_l1_loss: {gen_l1_loss}, disc_loss:{disc_loss}')
        gen_total_loss_val, gen_gan_loss_val, gen_l1_loss_val, disc_loss_val = self.evaluate_loss(self.val_ds)
        print(  f'Processing Epochs... Line 2 Validation Loss--> Epoch #: {epoch}, Experiment_ID: {self.experiment_id},gen_total_loss_val: {gen_total_loss_val}, gen_gan_loss_val: {gen_gan_loss_val}, gen_l1_loss_val: {gen_l1_loss_val}, disc_loss_val:{disc_loss_val}')

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
            gen_output = get_prediction(self.generator, input_image, target)
            self.writeMonitorImage((input_image, target, gen_output), f'{epoch}_{sample}_{i}')
        
    def config_modelSave(self):
        self.modelSave=f"{self.output_dir}/saved_Models/"
        self.create_dir(self.modelSave)
    
    def saveModel(self, epoch):
        k.models.save_model(self.generator, os.path.join(self.modelSave, f'{self.experiment_id}_{epoch}_old.h5'), save_format='h5')
        self.generator.save(os.path.join(self.modelSave, f'{self.experiment_id}_{epoch}.h5'))
        self.create_dir(os.path.join(self.modelSave, f'{self.experiment_id}_{epoch}_saved_model'))
        tf.saved_model.save( self.generator, os.path.join(self.modelSave, f'{self.experiment_id}_{epoch}_saved_model') )

    def fit(self, train_ds, val_ds, monitor_ds, epochs, experiment_id, dataroot, monitor_freq='No Monitoring', checkpointfreq=5, modelsavefreq = 5):

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

        batches = len(train_ds)
        print_freq = batches//10

        # Iterate Epochs
        for epoch in range(epochs):      
            start = time.time()
            print(f'Epoch Started .... Time: {start}, Epoch # {epoch}')

            # Iterate Batches
            for sample, (input_image, target) in train_ds.enumerate():
                gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = self.train_step(input_image, target)

                if sample%print_freq==0:
                    self.log_batches(sample, gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss)
                    if monitor_freq=='batch':
                        self.process_monitor(epoch, sample)
            
            # End of Epoch Reporting and Logging
            self.log_epochs(epoch, gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss)

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

        cv2.imwrite(f'{self.out_img_dir}/{fname}.png', gen_img)

    def processTestSample(self, input_image, target, image_file):
        gen_output = self.generator(input_image, training=False)
        def get_prediction(model, test_input, tar):   #MAY BE REMOVED 
            prediction = model(test_input, training=True)
            return prediction
        gen_output = get_prediction(self.generator, input_image, target)
        fname = str(image_file.numpy()[0]).replace("'", "").split('\\')[-1].replace('.png', '')
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

    def generator_loss(self, disc_generated_output, gen_output, target):

        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

        # Mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        total_gen_loss = gan_loss + (self.lamda * l1_loss)

        return total_gen_loss, gan_loss, l1_loss

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)

        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss
