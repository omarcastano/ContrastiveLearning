from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.metrics.cluster import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.neighbors import KNeighborsClassifier
from ContrastiveLearning.data_util import normalize_img
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier

import tensorflow as tf
batch_size =  32

#Projection Header
def get_projection_header(projected_dim):
    projection_h = tf.keras.models.Sequential([
        tf.keras.layers.Dense(projected_dim, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(projected_dim, activation='relu'),
        tf.keras.layers.BatchNormalization(),
    ], name='projection_header')
    return projection_h



#T-splot plot
def tsne_plot(dataset, encoder):
    data_labeled = (dataset
                .map(normalize_img, num_parallel_calls = tf.data.experimental.AUTOTUNE)
                .batch(batch_size)
                .prefetch(tf.data.experimental.AUTOTUNE))

    embeddings = encoder.predict(data_labeled)

    labeles_names = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
    labels =[]
    for i, l in dataset:
        labels.append(labeles_names[l.numpy()])

    tsne = TSNE(n_components = 2, perplexity=30, n_jobs = -1)
    #embeddings = MinMaxScaler().fit_transform(embeddings)
    img_tsne = tsne.fit_transform(embeddings)
    plt.figure(figsize=(11,7))
    sns.scatterplot(x=img_tsne[:,0], y=img_tsne[:,1], hue = labels, s=100) 
    plt.title('t-sne plot', fontsize=8)
    plt.show()
    
    
def KNN_test(ds_train, ds_test, encoder):
    #KNN
    data_train_labeled = (ds_train
                .map(normalize_img, num_parallel_calls = tf.data.experimental.AUTOTUNE)
                .batch(batch_size)
                .prefetch(tf.data.experimental.AUTOTUNE)
    )

    data_test_labeled = (ds_test
                .map(normalize_img, num_parallel_calls = tf.data.experimental.AUTOTUNE)
                .batch(batch_size)
                .prefetch(tf.data.experimental.AUTOTUNE)
    )

    labeles_names = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
    labels_train =[]
    for i, l in ds_train:
        labels_train.append(labeles_names[l.numpy()])

    labeles_names = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
    labels_test =[]
    for i, l in ds_test:
        labels_test.append(labeles_names[l.numpy()])

    embeddings_test = encoder.predict(data_test_labeled)
    embeddings_train = encoder.predict(data_train_labeled)


    scale = MinMaxScaler()
    embeddings_train = scale.fit_transform(embeddings_train)
    embeddings_test = scale.transform(embeddings_test)

    knn = KNeighborsClassifier(n_jobs = -1)
    knn.fit(embeddings_train, np.array(labels_train))

    print('\n-----------------------------------------------------')
    print('KNeighborsClassifier Results')
    print('Accuracy' , knn.score(embeddings_test, np.array(labels_test) ) )
    print('-----------------------------------------------------')
    return knn.score(embeddings_test, np.array(labels_test) )
