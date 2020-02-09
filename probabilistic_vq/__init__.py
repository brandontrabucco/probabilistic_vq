"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import tensorflow as tf


def get_logits(keys, embeddings):
    """Returns logits representing a pre-softmax distribution over the latent codes.
    
    Argruments:
    keys: float32 tensor with shape [batch_dim, num_channels]
    embeddings: float32 tensor with shape [num_codes, num_channels]
        
    Returns:
    float32 tensor with shape [batch_dim, num_codes]
    """
    return tf.reduce_mean(
        keys[..., tf.newaxis, :] - embeddings[..., :, tf.newaxis], axis=2)


def sample_from_logits(logits, embeddings):
    """Samples from embeddings using logits as a pre-softmax distribution
    
    Argruments:
    logits: float32 tensor with shape [batch_dim, num_codes]
    embeddings: float32 tensor with shape [num_codes, num_channels]
        
    Returns:
    float32 tensor with shape [batch_dim, num_channels]
    """
    return tf.gather(embeddings, tf.squeeze(
        tf.random.categorical(logits, 1), (-1)), axis=0)


def sample_best_from_logits(logits, embeddings):
    """Samples best embedding using logits as a pre-softmax distribution
    
    Argruments:
    logits: float32 tensor with shape [batch_dim, num_codes]
    embeddings: float32 tensor with shape [num_codes, num_channels]
        
    Returns:
    float32 tensor with shape [batch_dim, num_channels]
    """
    return tf.gather(embeddings, tf.squeeze(
        tf.math.argmax(logits, axis=1, output_type=tf.int32), (-1)), axis=0)


def pass_through_gradients(keys, samples):
    """Passes gradients through discrete latent embeddings.
    
    Argruments:
    keys: float32 tensor with shape [batch_dim, num_channels]
    samples: float32 tensor with shape [batch_dim, num_channels]
        
    Returns:
    float32 tensor with shape [batch_dim, num_channels]
    """
    return keys + tf.stop_gradient(samples - keys)


def get_kl_divergence(log_probs_a, log_probs_b):
    """Returns the kl divergence with the uniform distribution
    
    Argruments:
    log_probs_a: float32 tensor with shape [batch_dim, num_codes]
    log_probs_b: float32 tensor with shape [batch_dim, num_codes]
        
    Returns:
    float32 tensor with shape [batch_dim]
    """
    log_probs_ratio = log_probs_a - log_probs_b
    return tf.reduce_sum(tf.exp(log_probs_a) * log_probs_ratio, axis=(-1))
