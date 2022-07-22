import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime
from uuid import uuid4

from utils.datasets import load_configs
from utils.images import process_frame


parser = argparse.ArgumentParser()
parser.add_argument(
    '--config_dir',
    help='Dataset configuration file'
)
parser.add_argument(
    '--save_dir',
    help='Location to save the dataset; can be a previously collected dataset'
)

# Temporary measure, change later
image_config = {
    'target_size': (224, 224),
    'avoid_distortion': True
}


def collect_images():
    """
    Collect a batch of images from camera
    :return: images, labels
    """

    images_batch, labels_batch = [], []
    cv2.namedWindow('Collection')
    capture = cv2.VideoCapture(0)
    ret, frame = capture.read()

    while ret:
        cv2.imshow('collection', frame)
        ret, frame = capture.read()
        key = cv2.waitKey(20)

        # ESC is pressed, exit collection
        if key == 27:
            capture.release()
            cv2.destroyWindow('Collection')
            break

        # Backspace is pressed, delete last sample
        if key == 8:
            images_batch.pop()
            label = labels_batch.pop()
            class_count[class_map[label]] -= 1
            print('Class count:', class_count)

        # Collect gestures based on key press:
        for k, label in key_map.items():
            if key == k:
                image = process_frame(frame, **image_config)
                images_batch.append(image)
                labels_batch.append(label)
                class_count[class_map[label]] += 1
                print('Class count:', class_count)

    images_batch = np.array(images_batch)
    labels_batch = np.array(labels_batch)
    return images_batch, labels_batch


def save_images(images_batch, labels_batch):
    """
    Saves a batch of images according to their labels
    File names are created based on generated time and unique ID
    :param images_batch: batch of images
    :param labels_batch: batch of labels
    """
    print("Saving batch of images:")
    time_stamp = datetime.now().strftime('%Y%m-%d%H-%M-')
    for image, label in tqdm(zip(images_batch, labels_batch), total=len(images_batch), ncols=80):
        class_dir = os.path.join(save_dir, str(label))
        file_name = time_stamp + str(uuid4()) + '.jpg'
        # OpenCV saves image in BGR format
        image = image[:, :, ::-1]
        cv2.imwrite(os.path.join(class_dir, file_name), image)


if __name__ == '__main__':
    args = parser.parse_args()
    save_dir = args.save_dir

    # Load class and key maps
    class_map, key_map = load_configs(args.config_dir)
    class_count = {class_: 0 for class_ in class_map.values()}

    # Create save directory if it does not exist, count samples
    os.makedirs(save_dir, exist_ok=True)
    for class_ in class_map.keys():
        class_dir = os.path.join(save_dir, str(class_))
        os.makedirs(class_dir, exist_ok=True)
        class_count[class_map[class_]] = len(os.listdir(class_dir))
    print('Class count before collection:', class_count)

    # Collect and save images
    while True:
        images, labels = collect_images()
        save_images(images, labels)

        continue_ = input('Continue collection? [y/n] >> ')
        if continue_.lower() == 'n':
            break
