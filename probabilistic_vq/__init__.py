"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import tensorflow as tf


def negative_distance(x, y):
    """Returns the negative euclidean distance between two points
    in vector space

    Arguments:
    x: float32 tensor with shape [batch_dim, num_codes, num_channels]
    y: float32 tensor with shape [batch_dim, num_codes, num_channels]

    Returns:
    float32 tensor with shape [batch_dim, num_codes]
    """
    return -tf.keras.losses.mean_squared_error(x, y)


def anti_squashing_function(x):
    """Scales an input tensor x from the range (-inf, 0) to the new
    range (-inf, inf)

    Arguments:
    x: float32 tensor with shape [batch_dim, num_channels]

    Returns:
    float32 tensor with shape [batch_dim, num_codes]
    """
    return x - 1 / x


def anti_squash_negative_distance(x, y):
    """Returns the negative euclidean distance between two points
    in vector space mapped to the range (-inf, inf)

    Arguments:
    x: float32 tensor with shape [batch_dim, num_codes, num_channels]
    y: float32 tensor with shape [batch_dim, num_codes, num_channels]

    Returns:
    float32 tensor with shape [batch_dim, num_codes]
    """
    return -anti_squashing_function(
        tf.keras.losses.mean_squared_error(x, y))


def get_logits(keys,
               embeddings,
               similarity_method=anti_squash_negative_distance):
    """Returns logits representing a pre-softmax distribution over
    the latent codes
    
    Arguments:
    keys: float32 tensor with shape [batch_dim, num_channels]
    embeddings: float32 tensor with shape [num_codes, num_channels]
    similarity_method: a function that takes in two float32 tensors
        with shapes [batch_dim, num_codes, num_channels] and returns
        a float32 tensor with shape [batch_dim, num_codes]
        
    Returns:
    float32 tensor with shape [batch_dim, num_codes]
    """
    while len(embeddings.shape) < len(keys.shape) + 1:
        embeddings = embeddings[tf.newaxis]
    return similarity_method(
        keys[..., tf.newaxis, :], embeddings)


def sample_softly_from_logits(logits, embeddings):
    """Samples from embeddings using logits as a pre-softmax distribution
    by weighting each of the elements using attention

    Arguments:
    logits: float32 tensor with shape [batch_dim, num_codes]
    embeddings: float32 tensor with shape [num_codes, num_channels]

    Returns:
    float32 tensor with shape [batch_dim, num_channels]
    """
    while len(embeddings.shape) < len(logits.shape) + 1:
        embeddings = embeddings[tf.newaxis]
    return tf.reduce_sum(
        tf.math.softmax(logits)[..., tf.newaxis] * embeddings, axis=(-2))


def sample_randomly_from_logits(logits, embeddings):
    """Samples from embeddings using logits as a pre-softmax distribution
    by random picking an index
    
    Arguments:
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
    """Samples from embeddings using logits as a pre-softmax distribution
    by picking the best index
    
    Arguments:
    logits: float32 tensor with shape [batch_dim, num_codes]
    embeddings: float32 tensor with shape [num_codes, num_channels]
        
    Returns:
    float32 tensor with shape [batch_dim, num_channels]
    """
    a = tf.math.argmax(logits, axis=(-1), output_type=tf.int32)
    return tf.gather(embeddings, a, axis=0)


def kl_divergence(log_probs_a, log_probs_b):
    """Returns the kl divergence with the uniform distribution
    
    Arguments:
    log_probs_a: float32 tensor with shape [batch_dim, num_codes]
    log_probs_b: float32 tensor with shape [batch_dim, num_codes]
        
    Returns:
    float32 tensor with shape [batch_dim]
    """
    ratio = log_probs_a - log_probs_b
    return tf.reduce_sum(tf.exp(log_probs_a) * ratio, axis=(-1))


def pass_through(keys, samples):
    """Passes gradients through discrete latent embeddings.

    Arguments:
    keys: float32 tensor with shape [batch_dim, num_channels]
    samples: float32 tensor with shape [batch_dim, num_channels]

    Returns:
    float32 tensor with shape [batch_dim, num_channels]
    """
    return keys - tf.stop_gradient(keys) + samples
