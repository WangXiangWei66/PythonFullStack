import urllib.parse

kw = {'wd': '马士兵'}

result = urllib.parse.urlencode(kw)
print(result)

res = urllib.parse.unquote(result)
print(res)
