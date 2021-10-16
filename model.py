#Contrastive Learning model based on SIMCLR

import tensorflow as tf
from ContrastiveLearning.ContrastiveLoss import contrastive_loss

#contrastive model
class SimCLR(tf.keras.Model):
    def __init__(self, projected_dim, temperature):
        super(SimCLR, self).__init__()
        self.temperature = temperature
        
        self.encoder = get_encoder(0.0)
        self.projection_header = get_projection_header(projected_dim=projected_dim)
        print('Embedding dimension=' ,self.encoder.output.shape[1:], '\n')

        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.acc_metrics = tf.keras.metrics.CategoricalAccuracy(name='accuracy')
    def call(self,inputs):
        x = self.encoder(inputs)
        x = self.projection_header(x)
        return x

    @property
    def metrics(self):
        return [self.loss_tracker]
        
    def train_step(self, data):
        (x1, _) , (x2, _) = data

        with tf.GradientTape() as tape:
            z1 = self(x1, training=True)
            z2 = self(x2, training=True)
            #loss function
            loss = tf.reduce_mean(contrastive_loss(self.temperature)(z1, z2))

        #compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        

        #ppdate metrics and losses
        self.loss_tracker.update_state(loss)
        return {m.name: m.result() for m in self.metrics}

    

    def test_step(self, data):
        #augmented data
        (x1, _) , (x2, _) = data

        #get predictions to test
        z1 = self(x1, training=False)
        z2 = self(x2, training=False)

        #compute loss
        loss = tf.reduce_mean(contrastive_loss(self.temperature)(z1, z2))

        #update metrics and losses
        self.loss_tracker.update_state(loss)

        return {m.name: m.result() for m in self.metrics}
