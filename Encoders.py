import tensorflow as tf
from ContrastiveLearning.ResNet import ResNet18

def get_vgg_encoder(input_shape, dropout=None):
    encoder = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape) ,
        
        tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(dropout),

        tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal'), 
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(dropout),

        tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(), 
        tf.keras.layers.Dropout(dropout),

        tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.GlobalAveragePooling2D()
    ], name='Encoder')
    return encoder
  
  
def get_resnet18_encoder(input_shape):
    encoder = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape) ,
        ResNet18(input_shape)
    ], name='Encoder')
    return encoder

