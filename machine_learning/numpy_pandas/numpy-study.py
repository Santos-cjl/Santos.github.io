import numpy as np
#一、按Alt+shift+E可以把脚本文件同步到控制台
#1.创建数组
a1=np.array([1,2,3])

#二、创建矩阵
a2=np.zeros(shape=(3,4),dtype=int)
a3=np.ones(shape=(3,4),dtype=int)
a4=np.arange(1,5)
a5=np.random.randint(1,10,size=(3,4))

#三、caculate the shape and the size of an array
a5.shape
a5.size
a5.dtype

#四、矩阵的运算
a6=np.array([1,2,3,4,5])
a7=np.array([6,7,8,9,10])
a8=a6+a7
a9=a7-a6
a10=a6*a7

#五、sum\mean\max\min,etc...
a11=np.array([[1,2,3],[4,5,6]])
a12=np.sum(a11)
a13=np.mean(a11)    #求均值
a14=np.max(a11)
a15=np.median(a11)   #求中位数
a16=np.std(a11)    #求标准差
a17=np.var(a11)   #求方差

#六、矩阵的索引和切片
a18=np.array([[1,2,3],[4,5,6],[7,8,9]])
a18.argmax(axis=0)  #axis=0为列
a18.argmin(axis=1)  #axis=1为行
a18[0,1]
a18[0:2,1:3]
a18[:,1]

#七、矩阵的合并与拆分
a19=np.array([[1,2],[3,4]])
a20=np.array([[5,6],[7,8]])
a21=np.concatenate((a19,a20),axis=0)
a22=np.concatenate((a19,a20),axis=1)
a23=np.vstack((a19,a20))   #同a21
a24=np.hstack((a19,a20))   #同a22
a25=np.argmax(a20,axis=0)