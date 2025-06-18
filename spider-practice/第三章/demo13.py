import requests

url = 'https://www.baidu.com/img/PCtm_d9c8750bed0b3c7d089fa7d55720d6cf.png'

resp = requests.get(url)

#wb:以二进制写入的形式打开
with open('logo.png','wb') as file:
    #response.content是服务器返回的二进制内容
    file.write(resp.content)
