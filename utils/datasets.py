import os
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import json


def load_configs(config_dir):
    with open(config_dir, 'r') as f:
        maps = json.load(f)

    # Process class map
    class_map = maps['class_map']
    class_map = {int(k): v for k, v in class_map.items()}

    # Process key map
    key_map = maps['key_map']
    key_map = {ord(k): int(v) for k, v in key_map.items()}

    return class_map, key_map


def process_image(image, cvt_color=True, target_size=(224, 224, 3)):
    if image.shape != target_size:
        raise NotImplementedError
    if cvt_color:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_dataset(dataset_dir, recognizer, train=True):
    images, landmarks, labels = [], [], []
    undetected_count = defaultdict(lambda: 0)
    class_count = defaultdict(lambda: 0)

    # Test recognizer, trigger TensorFlow Lite message
    recognizer.image2vec(np.zeros((224, 224, 3), dtype='uint8'))

    print(f'Processing images from {len(os.listdir(dataset_dir))} classes:')
    for i, class_ in enumerate(os.listdir(dataset_dir)):
        for image_name in tqdm(os.listdir(os.path.join(dataset_dir, class_)), ncols=80):
            # Load and process image
            image = cv2.imread(os.path.join(dataset_dir, class_, image_name))
            image = process_image(image)
            lms = recognizer.image2vec(image, train)

            if lms is not None:
                class_count[i] += 1
                images.append(image)
                landmarks.append(lms)
                labels.append(i)
            else:
                undetected_count[i] += 1
                # print(os.path.join(class_, image_name), '- hand not detected')

    # Print messages
    if undetected_count:
        print('Undetected hands count by class:', dict(undetected_count))
    else:
        print('All hands are detected')
    print('Class count:', dict(class_count))

    return [np.asarray(x) for x in [images, landmarks, labels]]


def split_dataset(X, y, splits=(0.8, 0, 0.2), verbose=0):
    # Shuffle dataset
    assert len(X) == len(y)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

    # Split dataset
    assert sum(splits) == 1
    datasets, cur_idx = [], 0
    for split in splits:
        num_samples = int(split * len(X))
        indices = slice(cur_idx, cur_idx + num_samples)
        datasets.append([X[indices], y[indices]])
        cur_idx += num_samples

    # Print out splits information
    if verbose > 0:
        for i, dataset in enumerate(datasets):
            print(f'Dataset split {i + 1} size:', len(dataset[0]))

    return datasets
