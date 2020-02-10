"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from probabilistic_vq import get_logits
from probabilistic_vq import sample_softly_from_logits
from probabilistic_vq import sample_randomly_from_logits
from probabilistic_vq import sample_best_from_logits
import tensorflow as tf


class VectorQuantizer(tf.keras.layers.Layer):
    
    def __init__(self, num_codes, num_channels, **kwargs):
        """Builds a probabilistic vector quantization layer with num_codes 
        discrete latent variables in a num_channels size vector space
        
        Arguments:
        num_codes: int, the number of discrete latent variables
        num_channels: int, the number of channels in the layer output
        """
        tf.keras.layers.Layer.__init__(self, **kwargs)
        self.num_codes = num_codes
        self.num_channels = num_channels

        initializer = tf.keras.initializers.GlorotNormal()
        self.embeddings = tf.Variable(
            initializer([num_codes, num_channels]), trainable=True)

    def sample(self, logits, training=True, **kwargs):
        """Forward pass using probabilistic vector quantization
        in place of standard vector quantization

        Arguments:
        logits: float tensor with shape [batch_dim, num_codes]
        """
        if training:
            return sample_softly_from_logits(logits, self.embeddings)
        else:
            return sample_softly_from_logits(logits, self.embeddings)
        
    def call(self, keys, **kwargs):
        """Forward pass using probabilistic vector quantization
        in place of standard vector quantization
        
        Arguments:
        keys: float tensor with shape [batch_dim, num_channels]
        """
        return get_logits(keys, self.embeddings)
