import pandas as pd

data=[[1,'张三',20,98],
      [2,'李四',21,96],
      [3,'王五',20,94],
      [4,'陈六',19,93]]

column = ['学号','姓名','年龄','数学']
index = ['a','b','c','d']
# df = pd.DataFrame(data, columns=column)
df = pd.DataFrame(data, columns=column, index=index)
print(df)

#map使得不需要使用循环就可以达到批量处理效果
df['数学等级'] = df['数学'].map(lambda x:'优秀' if x > 95 else '良好')
print(df)