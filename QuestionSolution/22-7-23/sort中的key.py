lst = ['apple', 'orange','Pear','grape']

lst.sort()
print(lst)#按照英文字母排序，上面的为什么Pear排在前面，因为大写字母的ASCII码小

#忽略大小比较，这个时候就需要修改比较规则，通过key这个参数修改，key这个参数赋值的是一个函数
lst.sort(key = lambda x:x.upper())
print(lst)

#修改比较规则，按照英文字母的个数进行排序
lst.sort(key = lambda x:len(x))
print(lst)