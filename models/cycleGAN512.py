from models.cycleGANBase import cycleGANBase
from libs.generative.generators import unet_generator_512
from libs.generative.discriminators import patchgan_discriminator_512
import tensorflow as tf

OUTPUT_CHANNELS = 3

class cycleGAN512(cycleGANBase):

    def __init__(self,  **kwargs):
        cycleGANBase.__init__(self, **kwargs)

        self.generator_g = unet_generator_512(norm_type='instancenorm')
        self.generator_r = unet_generator_512(norm_type='instancenorm')

        self.discriminator_a = patchgan_discriminator_512(norm_type='instancenorm')
        self.discriminator_b = patchgan_discriminator_512(norm_type='instancenorm')

    def call(self, inputs):
        x2_ = self.generator_g(inputs)
        return x2_