###############################################
#Iris数据集，利用线性回归的办法得出其最小二乘解      #
#人工智能1903班                                 #
#王萱                                          #
#20195235                                     #
#copyright 2021                               #
###############################################

#######注意：测试时请将iris.xlsx转换为iris.xls格式，并与py文件放在同一目录！！！！####
import numpy as np
import xlrd

#打开xls文档
data = xlrd.open_workbook("iris.xls")  #将xlsx转换为xls便于操作
table = data.sheet_by_index(0)

one=1

temp=np.ones(3)
temp= np.matrix([table.cell_value(1 , 1), table.cell_value(1, 2), one]) # 第一行的特征向量
X=temp
for i in range(2 , 51):  #  对row进行遍历

    temp= np.matrix([table.cell_value(i , 1), table.cell_value(i, 2), one]) # 录入特征向量
    X=np.r_[X,temp] #特征矩阵
    print('\n')

print(X)
y=np.ones(1)
y=np.matrix([table.cell_value(1 , 3)]) # 第一行的标签向量
for i in range(2,51):
    y=np.r_[y,np.matrix([table.cell_value(i , 3)])]  # 录入标签向量

print(y)  #标签矩阵

X_T=X.transpose()
print(X_T)

Rx=np.dot(X_T,X)
print(Rx)  #自相关矩阵

Xy=np.dot(X_T,y)
print(Xy)  #互相关向量

w=np.dot(np.linalg.inv(Rx),Xy)
print(w)   #学习得出的最小二乘解


w_T=w.transpose()
loss=0
for i in range(1 , 51):  #  对row进行遍历

    temp= np.matrix([table.cell_value(i , 1), table.cell_value(i, 2), one]) # 录入特征向量
    a=np.dot(w_T,temp.transpose())
    y=table.cell_value(i , 3) # 录入标签向量
    p=y - a
    loss= p** 2 + loss

loss=loss/50
print(loss)