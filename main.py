import cv2
import time
from tqdm import tqdm


MODEL_FACE = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
MODEL_EYE = cv2.CascadeClassifier('models/haarcascade_eye.xml')
MODELS_PLATE = [
    cv2.CascadeClassifier(path) for path in (
        'models/haarcascade_licence_plate_rus_16stages.xml',
        'models/haarcascade_russian_plate_number.xml',
        )]


def main():

    # connect to camera
    camera = cv2.VideoCapture(0)
    while not camera.isOpened():
        time.sleep(0.2)

    # read and show frames
    try:
        with tqdm() as progress:
            while True:
                ret, frame = camera.read()
                cv2.imshow('Objects', process(frame))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                progress.update()
    finally:
        # gracefully close
        camera.release()
        cv2.destroyAllWindows()


def process(frame):
    """Process initial frame and tag recognized objects."""

    # 1. Convert initial frame to grayscale
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # For every model:
    for model, color, parameters in (
            (MODEL_FACE, (255, 255, 0), {'scaleFactor': 1.1, 'minNeighbors': 5, 'minSize': (30, 30)}),
            (MODEL_EYE, (0, 0, 255), {'scaleFactor': 1.1, 'minNeighbors': 5, 'minSize': (20, 20)}),
            *((model, (0, 255, 0), {'scaleFactor': 1.1, 'minNeighbors': 5, 'minSize': (20, 20)}) for model in MODELS_PLATE),
            ):
        # 2. Apply model, recognize objects
        objects = model.detectMultiScale(grayframe, **parameters)

        # 3. For every recognized object, draw a rectangle around it
        for (x, y, w, h) in objects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2) # BGR

    # 4. Return initial color frame with rectangles
    return frame


if __name__ == '__main__':
    main()
