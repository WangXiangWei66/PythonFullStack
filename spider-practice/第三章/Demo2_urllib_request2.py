import urllib.request
import urllib.parse

url = 'https://www.xslou.com/login.php'

#'action'对应的值为'login'，可能用于指示操作行为是登录。
data = {'username':'18600605736','password':'57365736','action':'login'}

resp = urllib.request.urlopen(url, data = bytes())