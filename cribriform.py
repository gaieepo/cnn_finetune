from __future__ import absolute_import

from sklearn.cross_validation import train_test_split
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

def convert_dataset(split_name, filenames, class_names_to_ids):
    """ Preprocess image to np array
    """
    assert split_name in ['train', 'test']
    data_list, labels_list = [], []
    for i in range(len(filenames)):
        # print("-> Converting %d/%d" % (i, len(filenames)))
        img = load_img(filenames[i][0], target_size=(256, 256, 3))

        data_list.append(img_to_array(img))
        # print("Size: " + str(img_to_array(img).shape))
        labels_list.append(filenames[i][1])

    data = np.stack(data_list)
    print(data.shape)
    labels = np.stack(labels_list)
    print(labels.shape)
    return data, labels


def load_data(fold):
    """ Load cribriform data from raw jpeg images
    
    use keras.preprosessing.image.[load_img and img_to_array]
    
    """

    dirname = '/diskb/tmp/gai/data_1106_256_180'
    train_positive = os.path.join(dirname, 'lists/list_fold_0%d_train_positive.txt' % fold)
    train_negative = os.path.join(dirname, 'lists/list_fold_0%d_train_negative.txt' % fold)
    test_positive = os.path.join(dirname, 'lists/list_fold_0%d_test_positive.txt' % fold)
    test_negative = os.path.join(dirname, 'lists/list_fold_0%d_test_negative.txt' % fold)

    with open(train_positive) as f:
        train_data_pos = [[l, 0] for l in f.readlines()]
    with open(train_negative) as f:
        train_data_neg = [[l, 1] for l in f.readlines()]
    with open(test_positive) as f:
        test_data_pos = [[l, 0] for l in f.readlines()]
    with open(test_negative) as f:
        test_data_neg = [[l, 1] for l in f.readlines()]

    train_data_pos.extend(train_data_neg)
    test_data_pos.extend(test_data_neg)

    for i in range(len(train_data_pos)):
        img_path = os.path.join(dirname, train_data_pos[i][0]).strip()
        train_data_pos[i][0] = img_path
    for i in range(len(test_data_pos)):
        img_path = os.path.join(dirname, test_data_pos[i][0]).strip()
        test_data_pos[i][0] = img_path

    class_names = ['Positives', 'Negatives']

    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    training_filenames = train_data_pos
    testing_filenames = test_data_pos

    print("#Training:"+str(len(training_filenames)))
    print("#Testing:"+str(len(testing_filenames)))

    # convert the training and test sets
    x_train, y_train = convert_dataset('train', training_filenames, class_names_to_ids)
    x_test, y_test = convert_dataset('test', testing_filenames, class_names_to_ids)

    # write the labels file
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))

    # (-1, 256, 256, 3)
    # (-1,)
    return (x_train, y_train), (x_test, y_test)
