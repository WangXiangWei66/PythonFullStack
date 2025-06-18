from pyquery import PyQuery

doc = PyQuery(filename='demo.html')
print(doc)
print(doc('h1'))
