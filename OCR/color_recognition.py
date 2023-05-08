import cv2

def color_recognition(image):
    """
    Эта функция определяет цвет фона бегущей строки
    :return: цвет фона
    """
    # Вычисление среднего цвета пикселей в области изображения
    average_color = cv2.mean(image)
    # Извлечение значений среднего цвета в формате BGR
    r, g, b = average_color[:3]

    # Определение пороговых значений RGB-компонент для синего, желтого и серого (белого) цветов
    blue_threshold = 200
    yellow_threshold = 220
    gray_threshold = 170

    # Определение функции для определения названия цвета на основе RGB-значений
    if r < blue_threshold and g > blue_threshold and b > blue_threshold:
        return "Синий"
    elif r > yellow_threshold and g > yellow_threshold and b < yellow_threshold:
        return "Желтый"
    elif r > gray_threshold and g > gray_threshold and b > gray_threshold:
        return "Серый (белый)"
    else:
        return "Неизвестный цвет"

