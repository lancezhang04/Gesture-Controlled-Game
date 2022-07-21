from utils.models import GestureRecognizer
from utils.datasets import load_dataset, split_dataset, load_configs
import pickle
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset_dir', default='images/left_neutral_right',
    help='The directory in which images are stored'
)
parser.add_argument(
    '--config_dir', default='configs/left_neutral_right_config.json',
    help='Configuration file to use for training'
)
parser.add_argument('--model_save_dir', default=None, help='Where to save the trained model')
parser.add_argument('--separate_test_dataset', default=None)
parser.add_argument('--predict_video_stream', action='store_true')


if __name__ == '__main__':
    args = parser.parse_args()
    # Check if `model_save_dir` is in correct format (if specified)
    assert args.model_save_dir.endswith('.pkl') if args.model_save_dir is not None else True

    # Load configs for control scheme
    class_map, key_map = load_configs(args.config_dir)
    # Load model
    recognizer = GestureRecognizer(class_map)

    # Load dataset, including predicting landmarks using mediapipe
    images, landmarks, labels = load_dataset(args.dataset_dir, recognizer)
    # Split dataset
    datasets = split_dataset(
        landmarks,
        labels,
        splits=[0.8, 0.2],
        verbose=1
    )

    # Fit and evaluate model
    print('\nTraining model...', end=' ')
    recognizer.clf.fit(*datasets[0])
    print('Complete\nTest accuracy:', recognizer.clf.score(*datasets[-1]))

    # Save trained model
    if args.model_save_dir is not None:
        with open(args.model_save_dir, 'wb') as f:
            pickle.dump(recognizer.clf, f)
        print('Model saved at', args.model_save_dir)
    else:
        print('No `model_save_dir` specified, model is not saved')

    # Load and split dataset from a new domain (unseen room)
    if args.separate_test_dataset is not None:
        images_test, landmarks_test, labels_test = load_dataset(
            args.separate_test_dataset,
            recognizer,
            train=False
        )
        dataset_test = split_dataset(
            landmarks_test,
            labels_test,
            splits=[1.0]
        )[0]
        print('Test (new domain) accuracy:', recognizer.clf.score(*dataset_test))

    # Predict from video stream
    if args.predict_video_stream:
        recognizer.predict_video_stream(
            continuous=True,
            plot_image=False,
            delay=1,
            show_capture=False
        )
