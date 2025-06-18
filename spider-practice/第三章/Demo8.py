import urllib.request
#error包含了在进行URL请求时可能引发的各种异常类
import urllib.error

# url = 'http://www.google.com'

url = 'http://www.google.cn'

try:
    resp = urllib.request.urlopen(url)

except urllib.error.URLError as e:
    print(e.reason)