import cv2
import time


FACE_MODEL_FILE = 'models/haarcascade_frontalface_default.xml'
EYES_MODEL_FILE = 'models/haarcascade_eye.xml'

PLATE_FILES = [
    'models/haarcascade_licence_plate_rus_16stages.xml',
    'models/haarcascade_russian_plate_number.xml',
    ]


def main():

    # load haar cascades model
    faces = cv2.CascadeClassifier(FACE_MODEL_FILE)
    eyes = cv2.CascadeClassifier(EYES_MODEL_FILE)
    plates = [cv2.CascadeClassifier(p) for p in PLATE_FILES]

    # connect to camera
    camera = cv2.VideoCapture(0)
    while not camera.isOpened():
        time.sleep(0.2)

    # read and show frames
    while True:

        ret, frame = camera.read()
        frame = process(frame, [
            (faces, (255, 255, 0), dict(scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))),
            (eyes, (0, 0, 255), dict(scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))),
            ])
        frame = process(frame, [
            (model, (0, 255, 0), dict(scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)))
            for model in plates
            ])
        cv2.imshow('Objects', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # gracefully close
    camera.release()
    cv2.destroyWindows()


def process(frame, models):
    """Process initial frame and tag recognized objects."""

    # 1. Convert initial frame to grayscale
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for model, color, parameters in models:

        # 2. Apply model, recognize objects
        objects = model.detectMultiScale(grayframe, **parameters)

        # 3. For every recognized object, draw a rectangle around it
        for (x, y, w, h) in objects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2) # BGR

    # 4. Return initial color frame with rectangles
    return frame



if __name__ == '__main__':
    main()
