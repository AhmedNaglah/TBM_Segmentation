from tensorflow.keras import Model
import tensorflow as tf
import os
import logging
import sys
import cv2
from libs.misc.z_helpers_metric import * 
import tensorflow.keras as k

class Base(Model):
    def __init__ (self, **kwargs):
        Model.__init__(self, **kwargs)
        self.loggers = {}

    def initiate_logger(self, logger_name, header, logfile):
        if logger_name in self.loggers:
            return False
        else:
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setLevel(logging.DEBUG)
            stdout_handler.setFormatter(formatter)

            file_handler = logging.FileHandler(logfile)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)

            logger.addHandler(file_handler)
            logger.addHandler(stdout_handler)
            stdout_handler = logging.StreamHandler(sys.stdout)
            logger.info(header)
            self.loggers[logger_name] = logger
            return True

    def write_log(self, logger_name, msg):
        self.loggers[logger_name].info(msg)

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

