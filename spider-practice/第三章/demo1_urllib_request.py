#url用于打开和读取url资源
import urllib.request
#chardet : 自动检测文本的编码格式
import chardet  # 新增库

url = 'https://www.aizhan.com/'
resp = urllib.request.urlopen(url)
content = resp.read()

# 自动检测编码
#[encoding]从字典中提取检测到的编码信息
encoding = chardet.detect(content)['encoding']
html = content.decode(encoding)

print(html)