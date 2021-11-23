import tensorflow as tf
import matplotlib.pyplot as plt

#normalize image
def normalize_img(img, label):
    #tf.cast cahnge de dtype of our tensor
    return tf.cast(img, dtype=tf.float32)/255.0 , label
  
  
#Random crop and horizontal flip
def random_crop_and_flip(image, croped_size):
    
    img_w, img_h, img_c = image.shape
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_crop(image, size=(croped_size, croped_size, 3))
    image = image = tf.image.resize(image, size=[img_w, img_h],  antialias=True, method='bilinear')
    #tf.image.resize_with_pad(image, img_w, img_h,  antialias=True)
    image = tf.clip_by_value(image, 0, 1)
    return image
  
# Color distortion
def random_color_distorio(image, strength = 1.0):

    #Random jitter
    def color_jitter(x):
        x = tf.image.random_brightness(x, max_delta=0.8*strength)
        x = tf.image.random_contrast(x, lower = 1 - 0.8*strength, upper=1 + 0.8*strength)
        x = tf.image.random_saturation(x, lower = 1 - 0.8*strength, upper=1 + 0.8*strength)
        x = tf.image.random_hue(x, max_delta=0.2*strength)
        x = tf.clip_by_value(x, 0, 1)

        return x

    #Color Drop
    def color_drop(x):
        x = tf.image.rgb_to_grayscale(x)
        x = tf.tile(x, [1, 1, 3])

        return x
    def random_apply(f, x, p):
        if tf.random.uniform([], minval=0, maxval=1) < p:
            return f(x)
        else: 
            return x

    image = random_apply(color_jitter, image,   p = 0.8)
    image = random_apply(color_drop, image, p = 0.2)
    return image
  
#data augmentation using random crop/flip and color distortion
def augmentation(image, label, strength=0.1, croped_size=26):
    image = random_crop_and_flip(image, croped_size=croped_size)
    image = random_color_distorio(image, strength=strength)
    return image, label
  
#visualize data augmentation
def visualize_augmentations(dataset, n_images, strength=0.1, croped_size=26):
    fig, ax  = plt.subplots(2,n_images, figsize=(n_images*3,6))
    for i, img in enumerate(dataset.take(n_images)):
        img = tf.cast(img[0], dtype=tf.float32)/255.0
        ax[0,i].imshow(img)
        ax[0,i].set_title('Original')
        img, _ = augmentation(img,-1, strength=strength, croped_size=croped_size)
        ax[1,i].imshow(img)
        ax[1,i].set_title('Augmented')
        ax[1,i].axis('off')
        ax[0,i].axis('off')


#get positive pairs        
def get_augmented_datasets(ds_unlabeled, batch_size, strength, croped_size):
    x1_train =  (ds_unlabeled.shuffle(600000, seed=26)
                .map(normalize_img, num_parallel_calls = tf.data.experimental.AUTOTUNE)
                .map(lambda x,y: augmentation(x,y,strength=strength,croped_size=croped_size) ,num_parallel_calls = tf.data.experimental.AUTOTUNE)
                #.cache()
                .batch(batch_size)
                .prefetch(tf.data.experimental.AUTOTUNE)
            )


    x2_train =  (ds_unlabeled.shuffle(600000, seed=26)
                .map(normalize_img, num_parallel_calls = tf.data.experimental.AUTOTUNE)
                .map(lambda x,y: augmentation(x,y,strength=strength,croped_size=croped_size) ,num_parallel_calls = tf.data.experimental.AUTOTUNE)
                #.cache()
                .batch(batch_size)
                .prefetch(tf.data.experimental.AUTOTUNE)
            )
    
    x = (ds_unlabeled.shuffle(600000, seed=26)
            .map(normalize_img, num_parallel_calls = tf.data.experimental.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

    data_unlabeled = tf.data.Dataset.zip((x1_train, x2_train))

    return x, data_unlabeled


def plot_augmented_data(data_unlabeled, x):
    (x1_view, _), (x2_view, _) = next(iter(data_unlabeled))
    x_view, _ = next(iter(x))

    fig, ax = plt.subplots(3, 10, figsize=(28,7))

    for i, j in enumerate(x_view[:10]):
        ax[0, i].imshow(j)
        ax[0, i].axis('off')
        ax[0, i].set_title('Original')

    for i, j in enumerate(x1_view[:10]):
        ax[1, i].imshow(j)
        ax[1, i].axis('off')
        ax[1, i].set_title('Augmentated 1')
        
    for i, j in enumerate(x2_view[:10]):
        ax[2, i].imshow(j)
        ax[2, i].axis('off')
        ax[2, i].set_title('Augmentated 2')

    plt.show()

