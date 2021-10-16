import tensorflow as tf

#Projection Header
def get_projection_header(projected_dim):
    projection_h = tf.keras.models.Sequential([
        tf.keras.layers.Dense(projected_dim, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(projected_dim, activation='relu'),
        tf.keras.layers.BatchNormalization(),
    ], name='projection_header')
    return projection_h
