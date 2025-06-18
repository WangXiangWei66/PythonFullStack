import requests
import json
import csv


def get_page():
    # 可能的API接口（需要抓包分析）
    api_url = "https://u.y.qq.com/cgi-bin/musicu.fcg"

    # 构造请求参数（根据实际API调整）
    params = {
        "data": json.dumps({
            "topList": {
                "module": "musicToplist.ToplistInfoServer",
                "method": "GetDetail",
                "param": {"topId": 4, "offset": 0, "num": 100}
            }
        })
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36",
        "Referer": "https://y.qq.com/n/ryqq/toplist/4"
    }

    response = requests.get(api_url, params=params, headers=headers)

    if response.status_code == 200:
        data = response.json()
        parse_api_data(data)


def parse_api_data(data):
    # 根据实际API返回的JSON结构解析
    # 这里需要根据实际返回的数据结构调整
    songs = data.get("topList", {}).get("data", {}).get("songInfoList", [])

    for song in songs:
        song_data = {
            "排名": song.get("rank"),
            "歌名": song.get("name"),
            "歌手": song.get("singer", [{}])[0].get("name"),
            "播放时间": song.get("interval")  # 可能需要转换为分钟:秒格式
        }
        saving_song(song_data)

# saving_song 函数保持不变
def saving_song(song):
    with open('Song.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([song.get('排名'), song.get('歌名'), song.get('歌手'), song.get('播放时间')])
