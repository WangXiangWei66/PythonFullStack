#requests: 发送http请求，轻松于Web服务器交互
import requests
#将 Python 数据结构（如字典、列表）转换为 JSON 格式的字符串（序列化)
import json

url = "https://api.96open.com/v1/chat/completions"

#json.dumps : 将python字典转化为json格式的字符串
request_data = json.dumps({
    "messages": [
        {
            "role": "system",
            "content": "你是一个大语言模型机器人"
        },
        {
            "role": "user",
            "content": "告诉我2022年的美国GDP是多少？"
        }
    ],
    #stream=False表示不是一个字一个字的往外输出
    "stream": False,
    "model": "gpt-4o-mini",
    "temperature": 0.5,
    "presence_penalty": 0.2,
    "frequency_penalty": 0.2,
    #top_p：选择累积概率达到top_p的词
    "top_p": 1
})

#headers:用于在http请求中设置头部信息
headers = {
    #Authorization:批准、授权
    "Authorization": "Bearer sk-efzt4FNH5vJdMRvv7HDJqEW0ScZdeiIlmq8yDKCaHk9COerB",
    #Content-Type:设置了请求体的数据格式
    "Content-Type": "application/json"
}

#POST 方法主要用于向服务器提交数据，以创建新的资源或执行特定的操作。
response = requests.request("POST", url, headers=headers, data=request_data)

print(response.text)
