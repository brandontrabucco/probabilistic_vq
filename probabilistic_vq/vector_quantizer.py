"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from probabilistic_vq import get_logits
from probabilistic_vq import sample_from_logits
from probabilistic_vq import sample_best_from_logits
from probabilistic_vq import pass_through_gradients
from probabilistic_vq import get_kl_divergence
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
        init = tf.keras.initializers.GlorotNormal()([num_codes, num_channels])
        self.embeddings = tf.Variable(init, trainable=True)
        
    def call(self, keys, training=True, **kwargs):
        """Forward pass using probabilistic vector quantization
        in place of standard vector quantization
        
        Arguments:
        keys: float tensor with shape [batch_dim, num_channels]
        """
        logits = get_logits(keys, self.embeddings)
        if training:
            samples = sample_from_logits(logits, self.embeddings)
        else:
            samples = sample_best_from_logits(logits, self.embeddings)
        samples = pass_through_gradients(keys, samples)
        log_probs = tf.math.log_softmax(logits, axis=(-1))
        return samples, log_probs
