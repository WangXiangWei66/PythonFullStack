import re
s = 'Istudy study Python3.7.9 every day'
print("----------match方法，从起始位置开始开始匹配")
#.group()用于获取匹配到的内容
print(re.match('I',s).group())
#字符串前加 r，表示原始字符串，忽略 Python 的转义规则
#\w:匹配任意数字字母下划线
print(re.match(r'\w',s).group())
#.匹配任意字符
print(re.match('.',s).group())

print("----------search方法，从任意位置开始匹配，匹配第一个成功位置")
print(re.search('study',s).group())
print(re.search(r's\w',s).group())
print("-------findall方法，从任意位置开始匹配，匹配多个，最终以列表的形式返回")
print(re.findall('y',s))
print(re.findall('Python',s))
#\.表示匹配一个实际的点号
#\d表示匹配任意十进制数
print(re.findall(r'P\w+.\d',s))
#.+表示匹配.一次或多次
print(re.findall(r'P.+\d',s))
print("--------sub方法的使用，替换功能----------")
#re.sub会替换所有满足条件的值
print(re.sub('study','like',s))
# print(re.sub(r's\w+','like',s))
