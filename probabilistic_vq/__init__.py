"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import tensorflow as tf


def get_logits(keys, embeddings):
    """Returns logits representing a pre-softmax distribution over
    the latent codes
    
    Argruments:
    keys: float32 tensor with shape [batch_dim, num_channels]
    embeddings: float32 tensor with shape [num_codes, num_channels]
        
    Returns:
    float32 tensor with shape [batch_dim, num_codes]
    """
    keys = keys[..., tf.newaxis, :]
    while len(embeddings.shape) < len(keys.shape):
        embeddings = embeddings[tf.newaxis]
    return tf.reduce_mean(keys - embeddings, axis=(-1))


def sample_randomly_from_logits(logits, embeddings):
    """Samples from embeddings using logits as a pre-softmax distribution
    
    Argruments:
    logits: float32 tensor with shape [batch_dim, num_codes]
    embeddings: float32 tensor with shape [num_codes, num_channels]
        
    Returns:
    float32 tensor with shape [batch_dim, num_channels]
    """
    s = tf.shape(logits)[:-1]
    a = tf.reshape(logits, [tf.reduce_prod(s), tf.shape(logits)[-1]])
    a = tf.reshape(tf.random.categorical(a, 1), s)
    return tf.gather(embeddings, a, axis=0)


def sample_best_from_logits(logits, embeddings):
    """Samples best embedding using logits as a pre-softmax distribution
    
    Argruments:
    logits: float32 tensor with shape [batch_dim, num_codes]
    embeddings: float32 tensor with shape [num_codes, num_channels]
        
    Returns:
    float32 tensor with shape [batch_dim, num_channels]
    """
    a = tf.math.argmax(logits, axis=(-1), output_type=tf.int32)
    return tf.gather(embeddings, a, axis=0)


def kl_divergence(log_probs_a, log_probs_b):
    """Returns the kl divergence with the uniform distribution
    
    Argruments:
    log_probs_a: float32 tensor with shape [batch_dim, num_codes]
    log_probs_b: float32 tensor with shape [batch_dim, num_codes]
        
    Returns:
    float32 tensor with shape [batch_dim]
    """
    ratio = log_probs_a - log_probs_b
    return tf.reduce_sum(tf.exp(log_probs_a) * ratio, axis=(-1))


def pass_through(keys, samples):
    """Passes gradients through discrete latent embeddings.

    Argruments:
    keys: float32 tensor with shape [batch_dim, num_channels]
    samples: float32 tensor with shape [batch_dim, num_channels]

    Returns:
    float32 tensor with shape [batch_dim, num_channels]
    """
    return keys - tf.stop_gradient(keys) + samples
