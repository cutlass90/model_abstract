import numpy as np
import unittest
import tensorflow as tf

import tools
from model_abstract import Model

class ToolsTests(unittest.TestCase):

    def test_split_data_set(self):
        images = np.arange(200).reshape([100,2])
        labels = np.vstack([np.eye(10)]*10)
        class_distrib = {i:i+1 for i in range(10)}

        splited_images, splited_labels =  tools.split_data_set(class_distrib, images, labels)
        self.assertEqual(splited_images.shape, (55,2))
        self.assertEqual(splited_labels.shape, (55,10))
        for i,v in enumerate(np.sum(splited_labels,0)):
            self.assertEqual(class_distrib[i], v)
        class_distrib[0]=11
        with self.assertRaises(ValueError):
            tools.split_data_set(class_distrib, images, labels)


class ModelAbstractTests(unittest.TestCase):

    def test_get_metrics(self):
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        sess = tf.InteractiveSession()
        labels = tf.constant([[1, 0,  0], [0, 1, 0], [0,  0, 1]], dtype=tf.float32)
        logits = tf.constant([[1,-7, -5], [7,-1, 4], [-2, 2, 0]], dtype=tf.float32)
        y_pred = np.array(   [[1, 0,  0], [1, 0, 0], [0,  1, 0]])
        y_true = np.array(   [[1, 0,  0], [0, 1, 0], [0,  0, 1]])
        precision, recall, f1, accuracy = Model().get_metrics(labels, logits)
        precision_, recall_, f1_, accuracy_ = sess.run([precision, recall, f1, accuracy])
        sess.close()

        precision_t, recall_t, f1_t, _ = precision_recall_fscore_support(y_true, y_pred)
        accuracy_t =  accuracy_score(y_true, y_pred)

        for i in range(labels.shape[1]):
            self.assertAlmostEqual(precision_t[i], precision_[i], places=3)
            self.assertAlmostEqual(recall_t[i], recall_[i], places=3)
            self.assertAlmostEqual(f1_t[i], f1_[i], places=3)
        self.assertAlmostEqual(accuracy_t, accuracy_)


if __name__ == '__main__':
    unittest.main()




