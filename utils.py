import os
import sys
import traceback

import numpy as np


def create_dataset(data_path, test_ratio=0.2, hot_labels=False, max_classes=None):
    files, id2label = get_filename_and_class(data_path=data_path,max_classes=max_classes)
    np.random.shuffle(files)
    num_test = int(test_ratio * len(files))

    np.random.shuffle(files)
    train_files = files[num_test:]
    test_files = files[:num_test]

    train_data = []
    train_labels = []

    test_data = []
    test_labels = []

    num_classes = len(id2label)

    for f in train_files:
        try:
            train_data.append(f[0])
            if hot_labels:
                y = np.zeros(shape=num_classes, dtype=np.float32)
                y[int(f[1])] = 1.0
                train_labels.append(y)
            else:
                train_labels.append(int(f[1]))
        except Exception as _:
            traceback.print_exc(file=sys.stdout)
            continue

    for f in test_files:
        try:
            test_data.append(f[0])
            if hot_labels:
                y = np.zeros(shape=num_classes, dtype=np.float32)
                y[int(f[1])] = 1.0
                test_labels.append(y)
            else:
                test_labels.append(int(f[1]))
        except Exception as _:
            traceback.print_exc(file=sys.stdout)
            continue

    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

    if len(train_data) == 0:
        train_data = None
        train_labels = None
    if len(test_data) == 0:
        test_data = None
        test_labels = None
    return train_data, train_labels, test_data, test_labels, id2label


def get_filename_and_class(data_path, max_classes=0, min_samples_per_class=0):
    """Returns a list of filename and inferred class names.
  Args:
      :param data_path: A directory containing a set of subdirectories representing class names. Each subdirectory should contain PNG or JPG encoded images.
      :param min_samples_per_class:
    :param max_classes:
    data_path:
  Returns:
    A list of image file paths, relative to `data_path` and the list of
    subdirectories, representing class names.
  """
    folders = [name for name in os.listdir(data_path) if
               os.path.isdir(os.path.join(data_path, name))]

    if len(folders) == 0:
        raise ValueError(data_path + " does not contain valid sub directories.")
    directories = []
    for folder in folders:
        directories.append(os.path.join(data_path, folder))

    folders = sorted(folders)
    id2label = {}

    i = 0
    c = 0
    total_files = []
    for folder in folders:
        print (len(folder))
        dir = os.path.join(data_path, folder)
        files = os.listdir(dir)
        if min_samples_per_class > 0 and len(files) < min_samples_per_class:
            continue

        for file in files:
            path = os.path.join(dir, file)
            total_files.append([path, i])
        id2label[i] = folder
        i += 1

        if 0 < max_classes <= c+1:
            break
        c += 1

    return np.array(total_files), id2label