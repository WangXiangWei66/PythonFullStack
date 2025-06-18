##requests: 发送http请求，轻松于Web服务器交互
import requests
##将 Python 数据结构（如字典、列表）转换为 JSON 格式的字符串（序列化)
import json

url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
#json.dumps：形成json对象
request_data = json.dumps({
    "messages":[
        {
            "role":"user",
            "content":"告诉我美国的GDP是多少？"
        }
    ],
    "model":"glm-4",
    #temperature:影响生成文本的随机性，值越高生成结果越具多样性和创造性。
    "temperature":0.9
})

headers = {
    'Authorization':'Bearer e9a3fd4df9544e149918a11f426fb8c6.aT7ti3NevK9Zgusv',
    'Content-Type':'application/json'
}

##POST 方法主要用于向服务器提交数据，以创建新的资源或执行特定的操作。
response = requests.request("POST",url,headers = headers,data = request_data)

print(response.text)