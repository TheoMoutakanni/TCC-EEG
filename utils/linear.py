from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from .plot import multiline

from tqdm import tqdm


def plot_knn(train_data, target, test_data=None, test_target=None, cmap_values=None, colorbar=True, colorbar_title=None, n_neighbors=10, regression=False, verbose=0):
    if verbose:
        train_data = tqdm(train_data)

    if regression:
        score_fn = mean_absolute_error
        random_pred = [np.median(target)]*len(target)
        plt.ylabel('MAE')
    else:
        score_fn = balanced_accuracy_score
        random_pred = np.random.choice(target, len(target))
        plt.ylabel('balanced accuracy')

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
                score.append(score_fn(test_target[i][val_inds], knn.predict(test_data[i][val_inds])))
            else:
                score.append(score_fn(target[val_inds], knn.predict(embds[val_inds])))

        score_list.append(np.array(score))
        if cmap_values is None:
            plt.plot(np.linspace(0.005, 0.3, 20), score)

    if cmap_values is not None:
        ln = multiline([percentage_train]*len(score), score_list, cmap_values, cmap='viridis')
        if colorbar:
            cb = plt.colorbar(ln)
            if colorbar_title is not None:
                cb.set_label(colorbar_title)
    random_score = score_fn(random_pred, target)
    plt.axhline(y=random_score, color='r', linestyle='--')
    plt.xlabel('% of training samples')
