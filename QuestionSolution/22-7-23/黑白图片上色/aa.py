import base64#用于处理图像编码
import requests#用于发送HTTP请求
#设置百度AI图像上色的请求URL
request_url = "https://aip.baidubce.com/rest/2.0/image-process/v1/colourize"
#以二进制的模式打开本地图像文件
f = open('test.png','rb')
#读取文件内容，并使用base64编码，将图像数据转化为字符串格式，便于通过HTTP请求发送
img = base64.b64encode(f.read())
#创建请求参数，将编码后的图像数据作为image参数传递给API
params = {"image": img}

access_token = "24.353838fdaad5b8added48313c43dcec4.2592000.1752228598.282335-119205201"
#将访问令牌附加到请求URL中
request_url = request_url + "?access_token=" + access_token
#设置请求头，指定内容的请求格式为表单编码
headers = {"Content-Type": "application/x-www-form-urlencoded"}
response = requests.post(request_url, data = params, headers = headers)

# if response:
#     print(response.json())
img = base64.b64decode(response.json()['image'])
file = open('result.jpg', 'wb')
file.write(img)
file.close()