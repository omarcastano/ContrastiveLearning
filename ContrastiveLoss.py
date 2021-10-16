import tensorflow as tf

#Contrastive loss based on SIMCLR
def contrastive_loss(temperature=0.1):
    def loss(proj_1, proj_2):
        proj_1 = tf.math.l2_normalize(proj_1, axis=1)
        proj_2 = tf.math.l2_normalize(proj_2, axis=1)

        similarity_1_2 = tf.matmul(proj_1, proj_2, transpose_b=True)/temperature
        similarity_2_1 = tf.matmul(proj_2, proj_1, transpose_b=True)/temperature

        similarity_1_1= tf.matmul(proj_1, proj_1, transpose_b=True)/temperature
        similarity_2_2= tf.matmul(proj_2, proj_2, transpose_b=True)/temperature

        batch_size = tf.shape(proj_1)[0]
        contrastive_labels = tf.one_hot(tf.range(batch_size), 2*batch_size)
        masks = tf.one_hot(tf.range(batch_size), batch_size)


        loss_1 = tf.keras.losses.categorical_crossentropy(contrastive_labels, tf.concat([similarity_1_2, similarity_1_1 - masks*1e9], axis=1) , from_logits=True)
        loss_2 = tf.keras.losses.categorical_crossentropy(contrastive_labels, tf.concat([similarity_2_1, similarity_2_2 - masks*1e9], axis=1) , from_logits=True)

        return loss_1 + loss_2
    return loss
