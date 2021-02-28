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


def plot_embeddings(embds, colors, cmap, legend_dict=None):
    plt.figure(figsize=(14,12))
    scatter = plt.scatter(embds[:,0], embds[:,1], c=colors, alpha=0.3, s=6, cmap=cmap)
    if legend_dict is not None:
        legend = plt.legend(*scatter.legend_elements(prop="colors"),
                        loc="lower right", title=legend_dict['title'])
        for i in range(len(legend.get_texts())):
            legend.get_texts()[i].set_text(legend_dict['color_mapping'][i])
        plt.gca().add_artist(legend)
    plt.colorbar()


def get_sleep_stages(subjects, windows_dataset):
    sleep_stages = [x[1] for x in BaseConcatDataset([windows_dataset.split('subject')[s] for s in subjects])]
    return np.array(sleep_stages)


def get_ages(subjects, windows_dataset, info):
    ages = [i for s in subjects
        for i in [info[info['subject'] == int(s)]['age'].iloc[0]]*len(windows_dataset.split('subject')[s])]
    return np.array(ages)


def get_subjects(subjects, windows_dataset):
    subjects = [int(i) for s in subjects
        for i in [s]*len(windows_dataset.split('subject')[s])]
    return np.array(subjects)
