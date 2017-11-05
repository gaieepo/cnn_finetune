from __future__ import absolute_import

from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

def convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir):
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


def load_data(dataset_dir='/diskb/tmp/gai/cribriform_256_fold_04_20170523'):
    """ Load cribriform data from raw jpeg images
    
    use keras.preprosessing.image.[load_img and img_to_array]
    
    """
    num_train_samples = 808 + 352 # pos+neg

    dirname = '/diskb/tmp/gai/0916_images'

    with open(os.path.join(dirname, 'lists/list_fold_04_train_positive.txt')) as f:
        train_data_pos = [[l, 0] for l in f.readlines()]
    with open(os.path.join(dirname, 'lists/list_fold_04_train_negative.txt')) as f:
        train_data_neg = [[l, 1] for l in f.readlines()]
    with open(os.path.join(dirname, 'lists/list_fold_04_test_positive.txt')) as f:
        test_data_pos = [[l, 0] for l in f.readlines()]
    with open(os.path.join(dirname, 'lists/list_fold_04_test_negative.txt')) as f:
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
    x_train, y_train = convert_dataset('train', training_filenames, class_names_to_ids, dataset_dir)
    x_test, y_test = convert_dataset('test', testing_filenames, class_names_to_ids, dataset_dir)

    # write the labels file
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))

    # (-1, 256, 256, 3)
    # (-1,)
    return (x_train, y_train), (x_test, y_test)
