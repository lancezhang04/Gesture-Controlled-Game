import cv2


def process_frame(frame, target_size=(224, 224), avoid_distortion=True):
    """
    Process video capture frame through cropping, flipping, and resizing
    :param frame: 2-D numpy array, frame captured from camera
    :param target_size: tuple, target size of output frame
    :param avoid_distortion: whether or not to avoid distortion by cropping
    :return: 2-D array, processed frame'
    """
    h, w, c = frame.shape

    # Up-scaling is probably not a good idea
    assert target_size[0] <= h
    assert target_size[1] <= w

    # Flip image horizontally and rearrange channels
    frame = frame[:, ::-1, ::-1]

    # Crop to avoid distortions
    if avoid_distortion:
        # Keeps the shorter dimension constant
        short_dim = 0 if h <= w else 1
        target_ratio = target_size[1 - short_dim] / target_size[short_dim]
        cur_ratio = frame.shape[1 - short_dim] / frame.shape[short_dim]
        ratio = target_ratio / cur_ratio

        crop_size = [0, 0]
        crop_size[short_dim] = frame.shape[short_dim]
        crop_size[1 - short_dim] = int(frame.shape[1 - short_dim] * ratio)
        crop_h, crop_w = crop_size

        # Perform center crop
        center = (h // 2, w // 2)
        x, y = center[1] - crop_w // 2, center[0] - crop_h // 2
        try:
            frame = frame[y: y + crop_h, x: x + crop_w, :]
        except Exception:
            raise ValueError('Target shape and frame dimensions are incompatible for no distortion')

    # Resize frame
    frame = cv2.resize(frame, target_size)

    return frame


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    capture = cv2.VideoCapture(0)
    ret, frame = capture.read()

    print('Original frame')
    plt.imshow(frame)
    plt.show()

    new_frame = process_frame(frame)
    plt.imshow(new_frame)
    plt.show()
