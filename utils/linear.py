from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


def plot_knn(embeds_list, target, names, verbose=0):
    if verbose:
        embeds_list = tqdm(embeds_list)
    for embds in embeds_list:
        score = []
        for per_train in np.linspace(0.005, 0.3, 20):
            knn = KNeighborsClassifier(n_neighbors=10)
            inds = np.arange(len(embds))
            np.random.shuffle(inds)
            train_inds, val_inds = inds[:int(
                per_train * len(embds))], inds[int(per_train * len(embds)):]
            knn.fit(embds[train_inds], target[train_inds])

            score.append(knn.score(embds[val_inds], target[val_inds]))

        plt.plot(np.linspace(0.005, 0.3, 20), score)

    random_score = accuracy_score(
    np.random.choice(target, len(target)), target)
    plt.axhline(y=random_score, color='r')
    plt.legend(names + ['random'])
    plt.xlabel('% of training samples')
    plt.ylabel('% of accuracy')
