import mediapipe as mp
from sklearn.svm import SVC
from sklearn.exceptions import NotFittedError
import matplotlib.pyplot as plt
import cv2


class GestureRecognizer:
    def __init__(self, class_map=None, saved_clf=None):
        # Initialize hand tracking models
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1
        )
        # During test, assume input to be video stream
        self.hands_test = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1
        )

        # Initialize classifier of choice -> SVC
        self.clf = saved_clf if saved_clf is not None else SVC(gamma=2, C=1)
        self.class_map = class_map

    def predict_landmarks(self, image, train=True):
        # Process an image using mediapipe to produce 63-d vector, image must be RGB
        if train:
            results = self.hands.process(image)
        else:
            results = self.hands_test.process(image)
        results = results.multi_hand_landmarks

        if not results:
            # print('No hand detected in image.')
            return None
        return results[0]

    def image2vec(self, image, train=True):
        # Process an image using mediapipe to produce 63-d vector, image must be RGB
        landmarks = self.predict_landmarks(image, train=train)
        if landmarks is None:
            return None

        # Create vector by concatenating coordinates of each landmark
        vec = []
        for lm in landmarks.landmark:
            coords = [lm.x, lm.y, lm.z]
            vec.extend(coords)
        return vec

    def predict_image(self, image):
        vec = self.image2vec(image)
        if vec is None:
            return None
        pred = self.clf.predict([vec])[0]
        return pred

    def predict_video_stream(self,
                             continuous=False,
                             plot_image=True,
                             show_capture=True,
                             delay=1):

        # Initialize window_size and video stream
        if show_capture:
            cv2.namedWindow('Test window_size')
        capture = cv2.VideoCapture(0)
        ret, frame = capture.read()

        # Key presses don't register when no window_size is initialized
        if not show_capture:
            continuous = True
        # Don't plot when continuously capturing
        if continuous:
            plot_image=False

        while ret:
            ret, frame = capture.read()
            if show_capture:
                cv2.imshow('Test window_size', frame)
            key = cv2.waitKey(delay)

            # Exit when ESC is pressed
            if key == 27:
                break
            # Capture when Enter is pressed
            if continuous or key == 13:
                frame = cv2.resize(frame, (224, 224))
                # Flip channels and horizontally
                frame = frame[:, ::-1, ::-1]

                pred = self.predict_image(frame)
                if pred is None:
                    pred = 'No hand detected'
                else:
                    pred = str(pred) if not self.class_map else self.class_map[pred]
                print(f'Model prediction: [{pred}]')

                if plot_image:
                    plt.axis('off')
                    plt.imshow(frame)
                    plt.title(f'Model prediction: [{pred}]')
                    plt.show()

        capture.release()
        if show_capture:
            cv2.destroyWindow('Test window_size')

    def plot_landmarks(self, image):
        # Visualize the image and its predicted landmarks
        landmarks = self.predict_landmarks(image)
        h, w, c = image.shape

        if landmarks is not None:
            # Make back up for model prediction
            image_bak = image.copy()

            # Visualize key points
            for lm in landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(image, (x, y), 3, (255, 0, 255), cv2.FILLED)

            # Visualize connections
            mp.solutions.drawing_utils.draw_landmarks(
                image, landmarks, mp.solutions.hands.HAND_CONNECTIONS
            )

            # Make prediction if model is fitted
            try:
                pred = self.clf.predict([self.image2vec(image_bak)])[0]
                title = str(pred) if self.class_map is None else self.class_map[pred]
            except NotFittedError as e:
                title = 'Model not fitted'

            # Show image, press ESC to exit
            cv2.imshow(title, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyWindow(title)
        else:
            print('No hand detected in image.')
