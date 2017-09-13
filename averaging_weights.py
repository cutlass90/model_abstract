import os
from pathlib import Path

import tensorflow as tf


def average_models(path_to_models, n_checkpoints, scope, save_path, verbose=False):
    """ Average last n models weights

        Args:
            path_to_models: str, path to folder with models
            n_checkpoints: int, number of models that will be averaged
            scope: str, variable scope name
            save_path: str, path to save averaged model
            vrbose: bool, verbose
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    path_to_checkpoint = path_to_models.joinpath('checkpoint')
    sess = tf.InteractiveSession()
    with path_to_checkpoint.open() as f:
        checkpoints = [os.path.join(str(path_to_models), line.split(' ')[1][1:-2])\
        for line in f if 'all_model_checkpoint_paths' in line]

    checkpoints = checkpoints[:n_checkpoints]
    if verbose: print('Checkpoints {} will be averaged '.format(checkpoints))

    saver = tf.train.import_meta_graph(checkpoints[0]+'.meta')
    for i, chp in enumerate(checkpoints):
        saver.restore(sess, chp)
        vars_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)
        names = [v.name for v in vars_]
        if i == 0:
            weights = sess.run(vars_)
        else:
            weights = [a+b for a, b in zip(weights, sess.run(vars_))]

    weights = [w/n_checkpoints for w in weights]
    assign_op = [tf.assign(v, w) for v, w in zip(vars_, weights)]
    sess.run(assign_op)
    saver.save(sess, save_path)

if __name__ == '__main__':
    # usage example
    path_to_models = Path('models/')
    n_checkpoints = 3
    scope = 'denoiser'
    save_path = 'models/averaged/averaged_weights'
    verbose = True

    average_models(path_to_models, n_checkpoints, scope, save_path, verbose)
