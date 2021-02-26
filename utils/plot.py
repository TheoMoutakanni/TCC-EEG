import colorsys

from braindecode.datasets.base import BaseConcatDataset
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sleep_cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', ['r', 'g', 'b', 'pink', 'orange'], len(['r', 'g', 'b', 'pink', 'orange']))
mapping = {  # We merge stages 3 and 4 following AASM standards.
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,
    'Sleep stage R': 4
}
reverse_mapping = {v: k[-1] for k,v in mapping.items()}

def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=True):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap


# Sleep stages

def plot_sleep_stages_pca(pca_embds, windows_dataset, subjects_dic):
    """
    Plot PCA of sleep stages using the embeddings.
    """
    sleep_stages = [x[1] for x in BaseConcatDataset([windows_dataset.split('subject')[s] for s in subjects_dic["self_valid_subjects"]])]
    plt.figure(figsize=(14,12))
    scatter = plt.scatter(pca_embds[:,0], pca_embds[:,1], c=sleep_stages, cmap=sleep_cmap, alpha=0.3, s=6)
    legend = plt.legend(*scatter.legend_elements(prop="colors"),
                        loc="lower right", title="Sleep stages")
    for i in range(len(legend.get_texts())):
        legend.get_texts()[i].set_text(reverse_mapping[i])
    plt.gca().add_artist(legend)
    plt.title("PCA of self-supervised embeddings according to sleep stages")


def plot_sleep_stages_tsne(tsne_embds, windows_dataset, subjects_dic):
    """
    Plot TSNE of sleep stages using the embeddings.
    """
    sleep_stages = [x[1] for x in BaseConcatDataset([windows_dataset.split('subject')[s] for s in subjects_dic["self_valid_subjects"]])]
    plt.figure(figsize=(14,12))
    scatter = plt.scatter(tsne_embds[:,0], tsne_embds[:,1], c=sleep_stages, cmap=sleep_cmap, alpha=0.3, s=6)
    legend = plt.legend(*scatter.legend_elements(prop="colors"),
                        loc="lower right", title="Sleep stages")
    for i in range(len(legend.get_texts())):
        legend.get_texts()[i].set_text(reverse_mapping[i])
    plt.gca().add_artist(legend)
    plt.title('TSNE of self-supervised embeddings according to sleep stages')


# Age

def plot_age_pca(pca_embds, windows_dataset, subjects_dic, info):
    """
    Plot PCA of patient ages using the embeddings.
    """
    ages = [i for s in subjects_dic["self_valid_subjects"] 
        for i in [info[info['subject'] == int(s)]['age'].iloc[0]]*len(windows_dataset.split('subject')[s])]
    plt.figure(figsize=(14,12))
    plt.scatter(pca_embds[:,0], pca_embds[:,1], c=ages, alpha=0.3, s=6, cmap=plt.get_cmap('inferno'))
    plt.colorbar()
    plt.title('PCA of self-supervised embeddings according to age')


def plot_age_tsne(tsne_embds, windows_dataset, subjects_dic, info):
    """
    Plot TSNE of patient ages using the embeddings.
    """
    ages = [i for s in subjects_dic["self_valid_subjects"] 
        for i in [info[info['subject'] == int(s)]['age'].iloc[0]]*len(windows_dataset.split('subject')[s])]
    plt.figure(figsize=(14,12))
    plt.scatter(tsne_embds[:,0], tsne_embds[:,1], c=ages, alpha=0.3, s=6, cmap=plt.get_cmap('inferno'))
    plt.colorbar()
    plt.title('TSNE of self-supervised embeddings according to age')


# Patient ID

def plot_patient_id_pca(pca_embds, windows_dataset, subjects_dic):
    """
    Plot PCA of patient ids using the embeddings.
    """
    subject = [int(i) for s in subjects_dic["self_valid_subjects"]
        for i in [s]*len(windows_dataset.split('subject')[s])]
    new_cmap = rand_cmap(
        max(subject)+1, type='bright', first_color_black=True,
        last_color_black=False, verbose=False)
    plt.figure(figsize=(14,12))
    plt.scatter(pca_embds[:,0], pca_embds[:,1], c=subject, alpha=0.3, s=6, cmap=new_cmap)
    plt.colorbar()
    plt.title('PCA of self-supervised embeddings according to patient id')
    plt.show()


def plot_patient_id_tsne(tsne_embds, windows_dataset, subjects_dic):
    """
    Plot TSNE of patient ids using the embeddings.
    """
    subject = [int(i) for s in subjects_dic["self_valid_subjects"]
        for i in [s]*len(windows_dataset.split('subject')[s])]
    new_cmap = rand_cmap(
        max(subject)+1, type='bright', first_color_black=True,
        last_color_black=False, verbose=False)
    plt.figure(figsize=(14,12))
    plt.scatter(tsne_embds[:,0], tsne_embds[:,1], c=subject, alpha=0.3, s=6, cmap=new_cmap)
    plt.colorbar()
    plt.title('TSNE of self-supervised embeddings according to patient id')
    plt.show()
