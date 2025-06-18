import re
import requests
import csv


def get_page():
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36'
    }
    response = requests.get('https://y.qq.com/n/ryqq/toplist/4', headers=headers)

    if response.status_code == 200:
        html = response.text
        parse_page(html)
    return None


def parse_page(html):
    pattern = re.compile(
        r'<li>.*?<div class="songlist__number.*?>(.*?)</div>.*?<div class="songlist__songname">.*?>(.*?)</div>.*?<div class="songlist__artist">.*?>(.*?)</div>.*?<div class="songlist__time">.*?>(.*?)</div>',
        re.S
    )
    items = re.findall(pattern, html)
    for item in items:
        song = {
            '排名': item[0],
            '歌名': item[1],
            '歌手': item[2],
            '播放时间': item[3]
        }
        saving_song(song)


def saving_song(song):
    with open('Song.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([song.get('排名'), song.get('歌名'), song.get('歌手'), song.get('播放时间')])


if __name__ == '__main__':
    get_page()
