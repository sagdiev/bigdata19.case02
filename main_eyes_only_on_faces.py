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

    # Создаем список распознанных объектов глаз
    for model, color, parameters in ((MODEL_EYE, (0, 0, 255), {'scaleFactor': 1.1, 'minNeighbors': 5, 'minSize': (20, 20)}),):

        objects_eyes = model.detectMultiScale(grayframe, **parameters)

    # Находим распознанные объекты лиц и проверяем приналежность глаз эти лицам
    for model, color, parameters in ((MODEL_FACE, (255, 255, 0), {'scaleFactor': 1.1, 'minNeighbors': 5, 'minSize': (30, 30)}),):

        objects_faces = model.detectMultiScale(grayframe, **parameters)

        # Для каждого распознанного объекта лица проверяем наличие глаз внутри прямоугольника
        for (x, y, w, h) in objects_faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)  # BGR
            for (x1, y1, w1, h1) in objects_eyes:
                if x1>x and y1>y and x1+w1<x+w and y1+h1<y+h:
                    cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 2)



    # 4. Return initial color frame with rectangles
    return frame


if __name__ == '__main__':
    main()
