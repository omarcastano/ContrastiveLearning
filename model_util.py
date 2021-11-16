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
import plotly.express as px
import pandas as pd
import pandas as pd

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
def tsne_plot(dataset, encoder, labeles_names, path):
    data_labeled = (dataset
                .map(normalize_img, num_parallel_calls = tf.data.experimental.AUTOTUNE)
                .batch(batch_size)
                .prefetch(tf.data.experimental.AUTOTUNE))

    embeddings = encoder.predict(data_labeled)


    labels =[]
    for i, l in dataset:
        labels.append(labeles_names[l.numpy()])

    tsne = TSNE(n_components = 2, perplexity=30, n_jobs = -1)
    #embeddings = MinMaxScaler().fit_transform(embeddings)
    img_tsne = tsne.fit_transform(embeddings)
    plt.figure(figsize=(11,7))
    sns.scatterplot(x=img_tsne[:,0], y=img_tsne[:,1], hue = labels, s=100) 
    plt.title('t-sne plot', fontsize=8)
    plt.savefig(path +'T-SNE.png')
    plt.show()

 #k nearest neighbor
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

    labels_train =[]
    for i, l in ds_train:
        labels_train.append(l.numpy())

    labels_test =[]
    for i, l in ds_test:
        labels_test.append(l.numpy())

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

#Neural Network at the top of the encoder
def ANN_test(ds_train, ds_test, input_shape, encoder, batch_size, epochs):
    data_train_labeled = (ds_train
                .map(normalize_img, num_parallel_calls = tf.data.experimental.AUTOTUNE)
                .shuffle(100000)
                .batch(batch_size)
                .prefetch(tf.data.experimental.AUTOTUNE)
    )

    data_test_labeled = (ds_test
                .map(normalize_img, num_parallel_calls = tf.data.experimental.AUTOTUNE)
                .shuffle(100000)
                .batch(batch_size)
                .prefetch(tf.data.experimental.AUTOTUNE)
    )

    encoder = encoder
    for layer in encoder.layers:
        layer.trainable = False

    print('\n-----------------------------------------------------')
    print(f'ANN Results (fine tune encoder = False )')

    #tf.keras.backend.clear_session()
    model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=[input_shape, input_shape, 3]),
            tf.keras.layers.RandomFlip(mode='horizontal'),
            tf.keras.layers.RandomContrast(factor=(0.1,0.5)),
            encoder,
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='nadam', loss="sparse_categorical_crossentropy", metrics = "acc")
    callback = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    model.fit(data_train_labeled, batch_size=batch_size, epochs=epochs, validation_data=data_test_labeled, callbacks=[callback], verbose=0)
    hist = pd.DataFrame(model.history.history)
    print(model.evaluate(data_test_labeled) )

    print(hist)
    fig = px.line(hist, y = ['loss', 'acc', 'val_loss', 'val_acc'], x = hist.index+1)
    fig = fig.update_layout(hovermode="x unified")
    fig.show()
    print('-------------------------------------------------------')
    result_1 = model.evaluate(data_test_labeled, verbose=0)



    for layer in model.layers:
        layer.trainable = True

    print('\n-----------------------------------------------------')
    print(f'ANN Results (fine tune encoder = True )')


    nadam = tf.keras.optimizers.Nadam(learning_rate=0.0001)
    model.compile(optimizer=nadam, loss="sparse_categorical_crossentropy", metrics = "acc")
    callback = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    model.fit(data_train_labeled, batch_size=batch_size, epochs=epochs, validation_data=data_test_labeled, callbacks=[callback], verbose=0)
    hist = pd.DataFrame(model.history.history)
    print(hist)
    print(model.evaluate(data_test_labeled) )

    
    fig = px.line(hist, y = ['loss', 'acc', 'val_loss', 'val_acc'], x = hist.index+1)
    fig = fig.update_layout(hovermode="x unified")
    fig.show()
    print('-------------------------------------------------------')
    result_2 = model.evaluate(data_test_labeled, verbose=0)
    return result_1, result_2



#Implements Kmeans given the pre-trained encoder network
def KMEANS_test(dataset, encoder):
    #KMEANS
    data_labeled = (dataset
                .map(normalize_img, num_parallel_calls = tf.data.experimental.AUTOTUNE)
                .batch(batch_size)
                .prefetch(tf.data.experimental.AUTOTUNE))

    embeddings = encoder.predict(data_labeled)

    labels =[]
    for i, l in dataset:
        labels.append(l.numpy())

    kmeans = KMeans(n_clusters = 10)
    y_pred = kmeans.fit_predict(embeddings)
    print('\n-----------------------------------------------------')
    print('Kmenas Clustering Results')
    print('adjusted_rand_score =' ,adjusted_rand_score(np.array(labels), y_pred))
    print('-----------------------------------------------------\n')
    return adjusted_rand_score(np.array(labels), y_pred)
