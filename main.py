import cv2
import numpy as np
from time import sleep


def task_1():
    img = cv2.imread('images/variant-4.png')
    # выделяем голубой канал
    b_chnl = img[:,:,0]
    # создаем пустое изображение (того же размера)
    b_img = np.zeros(img.shape)
    # записываем голубой канал
    b_img[:,:,0] = b_chnl
    # показываем результат
    cv2.imshow('blue channel', b_img)


def task_2():
    # подключаемся к камере
    cap = cv2.VideoCapture(0)
    # используемое разрешение
    resolution = (640, 480)
    # середина экрана относительно горизонтали
    middle_x = resolution[0] // 2
    for tick in range(120):
        sleep(0.5)  # в течение 120*0.5 = 60 секунд...
        # читаем фрейм
        ret, frame = cap.read()
        if not ret:
            continue  # повтор при некорректной попытке

        # приводим изображение к используемому разрешению
        frame = cv2.resize(frame, resolution, interpolation=cv2.INTER_LINEAR)
        # ищем метку по контурам в оттенках серого
        gray = cv2.GaussianBlur(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            (21, 21), 0
        )
        thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)[1]
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        # если контур(-ы) найдены
        if len(contours):
            # выбираем наибольший по площади
            c = max(contours, key=cv2.contourArea)
            # запоминаем размеры
            x, y, w, h = cv2.boundingRect(c)
            # если центр метки правее середины экрана
            if x + (w // 2) > middle_x:
                # выводим соответствующий текст
                cv2.putText(
                    frame, 'МЕТКА СПРАВА!', (middle_x, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, 2
                )

        # показываем результирующий фрейм
        cv2.imshow('Cam', frame)
    # отключаемся от камеры
    cap.release()


if __name__ == '__main__':
    task_1()
    # task_2()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
