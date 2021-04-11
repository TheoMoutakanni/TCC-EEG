from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from .plot import multiline

from tqdm import tqdm


def plot_knn(train_data, target, test_data=None, test_target=None, cmap_values=None, colorbar=True, colorbar_title=None, n_neighbors=10, regression=False, verbose=0):
    if verbose:
        train_data = tqdm(train_data)

    score_list = []
    percentage_train = np.linspace(0.005, 0.3, 20)
    for i, embds in enumerate(train_data):
        score = []
        for per_train in percentage_train:
            if not regression:
                knn = KNeighborsClassifier(n_neighbors=n_neighbors)
            else:
                knn = KNeighborsRegressor(n_neighbors=n_neighbors)
            inds = np.arange(len(embds))
            np.random.shuffle(inds)
            train_inds, val_inds = inds[:int(per_train * len(embds))], inds[int(per_train * len(embds)):]
            knn.fit(embds[train_inds], target[train_inds])

            if test_data is not None and test_target is not None:
                score.append(knn.score(test_data[i][val_inds], test_target[i][val_inds]))
            else:
                score.append(knn.score(embds[val_inds], target[val_inds]))

        score_list.append(np.array(score))
        if cmap_values is None:
            plt.plot(np.linspace(0.005, 0.3, 20), score)

    if cmap_values is not None:
        ln = multiline([percentage_train]*len(score), score_list, cmap_values, cmap='viridis')
        if colorbar:
            cb = plt.colorbar(ln)
            if colorbar_title is not None:
                cb.set_label(colorbar_title)

    if not regression:
        plt.ylabel('% of accuracy')
        random_score = accuracy_score(np.random.choice(target, len(target)), target)
        plt.axhline(y=random_score, color='r', linestyle='--')
    else:
        plt.ylabel('R2')

    plt.xlabel('% of training samples')
