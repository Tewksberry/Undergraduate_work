# -*- coding: utf-8 -*-
import datetime
import os
import subprocess
import time
import traceback
from datetime import timedelta

import cv2
import numpy as np
import pytesseract
import telebot
from PIL import Image
from dotenv import load_dotenv

from color_recognition import color_recognition


load_dotenv('.env')
BOT_TOKEN = os.environ.get('BOT_TOKEN')
CHAT_OCR_ID = os.environ.get('CHAT_OCR_ID')
CHAT_ERR_ID = os.environ.get('CHAT_ERR_ID')
bot = telebot.TeleBot(BOT_TOKEN)
print('bot is ready')

pytesseract.pytesseract.tesseract_cmd = r'D:\UserProfile\yosifovaae\scripts\ocr\tesseract\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = r'D:\UserProfile\yosifovaae\scripts\ocr\tesseract\tessdata'


def read_keywords(file_path):
    keywords = []
    with open(file_path, 'r') as file:
        for line in file:
            keywords.append(line.strip())
    return keywords


def text_recognition():
    """
    Эта функция получает и покадрово обрабатывает видеопоток, в случае распознавания в кадре отправляет
    сообщение в телеграм канал
    :return: нет
    """
    # Fetch keywords from txt file
    keywords = read_keywords(r'D:\UserProfile\yosifovaae\scripts\ocr\keywords.txt')
    for keyword in keywords:
        # Выполнение действий с каждым ключевым словом
        print(keyword)

    # Получение доступа к видеопотоку прямого эфира
    with open(r'D:\UserProfile\yosifovaae\scripts\face_recognition\news_url.txt', 'r') as f:
        news_url = f.read().strip()

    ffmpeg_cmd = ['ffmpeg', '-i', news_url, '-f', 'image2pipe', '-pix_fmt', 'rgb24', '-vf',
                  'scale=1280:720,setsar=1', '-vcodec', 'rawvideo', '-']
    ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE)
    frame_count = 0

    while True:
        frame_bytes = ffmpeg_process.stdout.read(1280*720*3)
        frame_count += 1
        frame_np = np.frombuffer(frame_bytes, np.uint8)
        if np.all(frame_np == 0):
            return
    
        frame = frame_np.reshape((720, 1280, 3))
        if frame_count % 30 != 0:  # обрабатываем только каждый 30-й кадр, можно менять
            continue

        # Обрезка кадра, оставляя только его четверть снизу
        frame_cut1 = frame[10*frame.shape[0]//11:frame.shape[0], 1*frame.shape[1]//15:frame.shape[1], :]
        frame_cut2 = frame[10*frame.shape[0]//11:frame.shape[0], 3*frame.shape[1]//13:frame.shape[1], :]

        # Обработка фрейма и распознавание
        frame_uncol = cv2.cvtColor(frame_cut2, cv2.COLOR_BGR2GRAY)
        result = pytesseract.image_to_string(frame_uncol, lang='rus')
        # Check for keywords in recognized text
        for keyword in keywords:
            if keyword.lower() in result.lower():
                image_name = (datetime.datetime.utcnow() + timedelta(hours=3)).strftime('%d-%m-%Y %H_%M_%S')
                current_time = (datetime.datetime.utcnow() + timedelta(hours=3)).strftime('%H:%M:%S')
                # Преобразование массива numpy в объект изображения
                image = Image.fromarray(frame_cut1)
                # Генерация имени файла на основе текущего времени
                filename = f'recognitions/image_{image_name}.jpg'
                # Сохранение изображения с динамически сгенерированным именем файла
                image.save(filename)
                print(f'Keyword found: {keyword}')
                color = color_recognition(frame_cut2)

                message = f"{result}" \
                          f"Время: {current_time}\n" \
                          f"Цвет: {color}"

                # telebot send message
                bot.send_photo(chat_id=CHAT_OCR_ID, photo=image, caption=message)

        time.sleep(5)
        print(result)


try:
    while True:
        # Получение доступа к видеопотоку прямого эфира
        with open(r'D:\UserProfile\yosifovaae\scripts\face_recognition\news_url.txt', 'r') as f:
            news_url = f.read().strip()
        text_recognition()

except Exception as e:
    error = traceback.format_exc(limit=3)
    text = 'Ошибка в работе OCR! Система отключена! @tewksberry'
    bot.send_message(chat_id=CHAT_ERR_ID, text=f"{text}\n\n{error}")
