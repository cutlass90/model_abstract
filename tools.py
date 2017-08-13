import numpy as np
from sklearn.model_selection import train_test_split


def split_data_set(class_distrib, images, labels, verbose=False):
    # class_distrib: dict, key - class from 0 to 9, item - number of samples
    def print_stat(labels):
        [print('class_{0} = {1}'.format(i,v)) for i,v in enumerate(np.sum(labels,0))]
    if verbose:
        print('Set summary:')
        print_stat(labels)
    for i,v in enumerate(np.sum(labels,0)):
        if class_distrib[i] > v:
            raise ValueError('There is not enough data for class {0},\
                request{1}, find {2}'.format(i, class_distrib[i], v))

    indxs = []
    for cl, n in class_distrib.items():
        indxs.append(np.random.permutation(np.nonzero(labels[:, cl] == 1)[0])[:n])
    indxs = np.concatenate(indxs)
    return images[indxs], labels[indxs]
