from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.metrics.cluster import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.neighbors import KNeighborsClassifier
from ContrastiveLearning.data_util import normalize_img

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



#T-splot plot
def tsne_plot(dataset, encoder):
    data_labeled = (dataset
                .map(normalize_img, num_parallel_calls = tf.data.experimental.AUTOTUNE)
                .batch(1)
                .prefetch(tf.data.experimental.AUTOTUNE))

    embeddings = encoder.predict(data_labeled)

    labeles_names = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
    labels =[]
    for i, l in dataset:
        labels.append(labeles_names[l.numpy()])

    tsne = TSNE(n_components = 2, perplexity=30, n_jobs = -1)
    embeddings = MinMaxScaler().fit_transform(embeddings)
    img_tsne = tsne.fit_transform(embeddings)
    plt.figure(figsize=(11,7))
    sns.scatterplot(x=img_tsne[:,0], y=img_tsne[:,1], hue = labels, s=100) 
    plt.title('t-sne plot', fontsize=8)
    plt.show()
