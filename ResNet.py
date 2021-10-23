import tensorflow as tf

class ResidualUnit(tf.keras.layers.Layer):

    def __init__(self, filters, activation, stride=1):
        super(ResidualUnit, self).__init__()
        self.conv1 =  tf.keras.layers.Conv2D(filters, kernel_size=3, strides=stride ,padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.activ = tf.keras.activations.get(activation)
        self.conv2 =  tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1 ,padding='same')
        self.bn2 =  tf.keras.layers.BatchNormalization()

        if stride > 1:
            self.conv3 = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same')
            self.bn3 = tf.keras.layers.BatchNormalization()
        else:
            self.conv3 = lambda x:x
            self.bn3 = lambda x:x

    def call(self, inputs):

        x = inputs
        x= self.conv1(x)
        x= self.bn1(x)
        x= self.activ(x)

        x= self.conv2(x)
        x= self.bn2(x)

        skip_x = inputs
        skip_x = self.conv3(skip_x)
        skip_x = self.bn3(skip_x)

        return self.activ(x + skip_x)
      
      
class ResNet18(tf.keras.Model):
    def __init__(self,input_shape, **kwargs):
        super(ResNet18, self).__init__(**kwargs)

        self.stem=tf.keras.models.Sequential([
            tf.keras.layers.Input(input_shape),
            tf.keras.layers.Conv2D(64, kernel_size=3 ,strides=1, input_shape=[32,32,3]),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(1,1),padding='same')
        ])

        self.res_64_1 = ResidualUnit(filters=64, activation='relu', stride=1)
        self.res_64_2 = ResidualUnit(filters=64, activation='relu', stride=1)

        self.res_128_1 = ResidualUnit(filters=128, activation='relu', stride=2)
        self.res_128_2 = ResidualUnit(filters=128, activation='relu', stride=1)

        self.res_256_1 = ResidualUnit(filters=256, activation='relu', stride=2)
        self.res_256_2 = ResidualUnit(filters=256, activation='relu', stride=1)

        self.res_512_1 = ResidualUnit(filters=512, activation='relu', stride=2)
        self.res_512_2 = ResidualUnit(filters=512, activation='relu', stride=1)

        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()


    def call(self, inputs):

        x = self.stem(inputs)

        x = self.res_64_1(x)
        x = self.res_64_2(x)

        x = self.res_128_1(x)
        x = self.res_128_2(x)

        x = self.res_256_1(x)
        x = self.res_256_2(x)

        x = self.res_512_1(x)
        x = self.res_512_2(x)

        x = self.avg_pool(x)

        return x
