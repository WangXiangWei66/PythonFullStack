#两个键值对 name 对应的是Python  address对应的是一个字典
#字典中也对应了两个键值对 province对应的是吉林省  键city对应了一个列表
my_json = {'name' : 'Python', 'address' : {'province' : '吉林省', 'city' : ['长春市', '吉林市', '松原市']}}

#获取吉林省

province = my_json['address']['province']
print(province)

#获取吉林市
city = my_json['address']['city'][1]
print(city)