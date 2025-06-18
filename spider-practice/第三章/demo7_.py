import urllib.request
#cookiejar:用于处理HTTP cookies

#Cookies 是服务器发送到用户浏览器并存储在本地的一小段数据，
# 用于识别用户身份、记录用户偏好等。
from http import cookiejar

filename = 'cookie.txt'

def get_cookie():

    #实例化一个MozillaCookieJar对象
    cookie = cookiejar.MozillaCookieJar(filename)
    #创建HTTPCookieProcessor处理器对象。用于处理HTTP请求中的cookies
    handler = urllib.request.HTTPCookieProcessor(cookie)
    #build_opener:用于构建OpenerDirect对象
    opener = urllib.request.build_opener(handler)
    url = 'https://tieba.baidu.com/index.html?traceid=#'
    resp = opener.open(url)
    #将cookie对象保存在指定的文件中
    cookie.save()

def use_cookie():
    cookie = cookiejar.MozillaCookieJar()
    cookie.load(filename)
    print(cookie)

if __name__ == '__main__':
    use_cookie()