from utils.models import GestureRecognizer
from utils.datasets import load_dataset, split_dataset

class_map = {
    0: 'thumb_left',
    1: 'thumb_right'
}
key_map = {
    ord('a'): 0,  # thumb pointing left
    ord('d'): 1  # thumb pointing right
}

# Load and split dataset
recognizer = GestureRecognizer(class_map)
images, landmarks, labels = load_dataset('images/left_right', recognizer)
datasets = split_dataset(landmarks, labels, splits=[0.8, 0.2])

# Fit and score model
recognizer.clf.fit(*datasets[0])
print('Test accuracy:', recognizer.clf.score(*datasets[1]))

# Load dataset from a new domain (unseen room)
images_test, landmarks_test, labels_test = load_dataset('images/left_right_test', recognizer, train=False)
dataset_test = split_dataset(landmarks_test, labels_test, splits=[1.0])[0]
print('Test (new domain) accuracy:', recognizer.clf.score(*dataset_test))

# Inspect individual image
# recognizer.plot_landmarks(images_test[5])

# Predict from video stream
recognizer.predict_video_stream(continuous=True, plot_image=False, delay=1, show_capture=False)

print('Complete.')
