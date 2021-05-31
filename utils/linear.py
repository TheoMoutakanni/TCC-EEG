from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from .plot import multiline

from tqdm.autonotebook import tqdm


def plot_knn(train_data, target, test_data=None, test_target=None, cmap_values=None, colorbar=True, colorbar_title=None, n_neighbors=10, regression=False, percentage_train=None, verbose=0):
    if regression:
        score_fn = mean_absolute_error
        random_pred = [np.median(target)]*len(target)
        plt.ylabel('MAE')
    else:
        score_fn = balanced_accuracy_score
        random_pred = np.random.choice(target, len(target))
        plt.ylabel('balanced accuracy')

    score_list = []
    if percentage_train is None:
        percentage_train = np.linspace(0.005, 0.3, 20)

    train_data = tqdm(train_data) if verbose else train_data

    for i, embds in enumerate(train_data):
        score = []
        percentage_train_it = tqdm(percentage_train, leave=False) if verbose else percentage_train
        for per_train in percentage_train_it:
            if not regression:
                knn = KNeighborsClassifier(n_neighbors=n_neighbors)
            else:
                knn = KNeighborsRegressor(n_neighbors=n_neighbors)
            inds = np.arange(len(embds))
            np.random.shuffle(inds)
            train_inds, val_inds = inds[:int(per_train * len(embds))], inds[int(per_train * len(embds)):]
            knn.fit(embds[train_inds], target[train_inds])

            if test_data is not None and test_target is not None:
                score.append(score_fn(test_target, knn.predict(test_data[i])))
            else:
                score.append(score_fn(target[val_inds], knn.predict(embds[val_inds])))

        score_list.append(np.array(score))
        if cmap_values is None:
            plt.plot(percentage_train, score)

    if cmap_values is not None:
        ln = multiline([percentage_train]*len(score), score_list, cmap_values, cmap='viridis')
        if colorbar:
            cb = plt.colorbar(ln)
            if colorbar_title is not None:
                cb.set_label(colorbar_title)
    random_score = score_fn(random_pred, target)
    plt.axhline(y=random_score, color='r', linestyle='--')
    plt.xlabel('% of training samples')


def plot_knn_noise(train_data, target, test_data, test_target, noise_levels, percentage_train=0.02, cmap_values=None, colorbar=True, colorbar_title=None, n_neighbors=10, regression=False, verbose=0):
    train_data = np.array(train_data)

    if regression:
        score_fn = mean_absolute_error
        random_pred = [np.median(target)]*len(target)
        plt.ylabel('MAE')
    else:
        score_fn = balanced_accuracy_score
        random_pred = np.random.choice(target, len(target))
        plt.ylabel('balanced accuracy')

    score_list = []

    inds = np.arange(len(target))
    np.random.shuffle(inds)
    train_inds = inds[:int(percentage_train * len(target))]

    train_data_it = tqdm(range(len(train_data))) if verbose else range(len(train_data))

    for i in train_data_it:
        score = []
        if not regression:
            knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        else:
            knn = KNeighborsRegressor(n_neighbors=n_neighbors)

        if len(train_data.shape) == 3:
            knn.fit(train_data[i][train_inds], target[train_inds])

        for j, noise_embds in enumerate(test_data[i]):
            if len(train_data.shape) == 4:
                knn.fit(train_data[i,j][train_inds], target[train_inds])
            score.append(score_fn(test_target, knn.predict(noise_embds)))

        score_list.append(np.array(score))
        if cmap_values is None:
            plt.plot(noise_levels, score)

    if cmap_values is not None:
        ln = multiline([noise_levels]*len(score), score_list, cmap_values, cmap='viridis')
        if colorbar:
            cb = plt.colorbar(ln)
            if colorbar_title is not None:
                cb.set_label(colorbar_title)
    random_score = score_fn(random_pred, target)
    plt.axhline(y=random_score, color='r', linestyle='--')
    plt.xlabel('noise std')


def compute_dist_to_knn(X, k='auto'):
    # Compute the distance for each point to their knn
    # k = 'auto' -> take 20% of total number of points
    if k == 'auto':
        k = int(len(X) / 5.)
    elif type(k) == float:
        k = int(len(X) * k)

    if k == 0:
        print("Warning, k==0")
        return [np.nan]

    neigh = NearestNeighbors(n_neighbors=k, n_jobs=-1)
    neigh.fit(X)
    dist, ind = neigh.kneighbors(X)
    return np.array([dist[i][k - 1] for i in range(len(dist))])