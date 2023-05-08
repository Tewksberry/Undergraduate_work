import os
import re
import traceback
from datetime import datetime
from typing import Optional, NamedTuple

import requests
import telebot
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv('.env')
BOT_TOKEN = os.environ.get('BOT_TOKEN')
CHAT_OCR_ID = os.environ.get('CHAT_OCR_ID')
CHAT_ERR_ID = os.environ.get('CHAT_ERR_ID')
bot = telebot.TeleBot(BOT_TOKEN)


class Answer(NamedTuple):
    status: bool
    result: Optional[str] = None


def update_news_url():
    """
    Эта функция получает токен через html код страницы и формирует ссылку на прямую трансляцию
    :return: news_url
    """
    url = "https://peers.tv/moskva_24/"
    response = requests.get(url)
    html = response.text
    soup = BeautifulSoup(html, "html.parser")
    script_tag = soup.find("script", text=re.compile("window.AUTH_TOKEN"))
    token_match = re.search(r'window.AUTH_TOKEN\s*=\s*\'(.+?)\'', script_tag.string)
    token_time = re.search(r'window.AUTH_EXPIRES\s*=\s*\'(.+?)\'', script_tag.string)
    api_cl = re.search(r'window.API_CLIENT\s*=\s*\'(.+?)\'', script_tag.string)

    if token_match:
        token = token_match.group(1)
        time_exp = token_time.group(1)
        time_obj = datetime.strptime(time_exp, '%Y-%m-%dT%H:%M:%S%z')
        time_str = time_obj.strftime('%d.%m.%Y %H:%M:%S %Z')
        client = api_cl.group(1)
        news_url = f'https://spb1.peers.tv/streaming/moskva_24/126/tvrecw/playlist.m3u8?token={token}&client=81'

        message = f"Ссылка на эфир новостей обновлена: {news_url}\n\nВремя истечения ссылки: {time_str}\n\n" \
                  f"Пользователь: {client}"
        bot.send_message(chat_id=CHAT_OCR_ID, text=message)
        print(message)
        
        # Записываем ссылку в файл news_url.txt
        with open("news_url.txt", "w") as f:
            f.write(news_url)

        return Answer(True, f"Ссылка на видеопоток обновлена!")

    else:
        text1 = 'Face recognition system: Значение токена не найдено!'
        bot.send_message(chat_id=CHAT_ERR_ID, text=text1)
        return Answer(False, text1)


if __name__ == '__main__':
    try:
        update_news_url()
    except Exception as e:
        error2 = traceback.format_exc(limit=3)
        text2 = 'Ошибка при обновлении ссылки на видеопоток!'
        bot.send_message(chat_id=CHAT_ERR_ID, text=f"{text2}\n\n{error2}")
