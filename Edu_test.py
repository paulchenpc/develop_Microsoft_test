#例5.4
import numpy as np
import pandas as pd
import math

data=pd.read_csv(r"F:\example5_4.csv",encoding='gbk')
data.columns = ['c', 'x1', 'x2']

g=data
lb=g.iloc[:,0]  #数据集的第一列"c"的值

a=[0,3.12,497]  #待归类的新申请者
tx=pd.DataFrame(a)
tx=tx.transpose()

def find_maxnum(a, b, c):
    maxnum = 0
    if a >= b:
        maxnum = a
        lb = 1
    else:
        maxnum = b
        lb = 2
    # 再比较maxnum和c
    if c > maxnum:
        maxnum = c
        lb = 3
    return maxnum, lb

def Bayes_method_of_two_populations(g,tx):
    x=g.iloc[:,1:3]
    label=np.array(g.iloc[:,0])

    data_g=g.groupby('c')
    g1=dict([x for x in data_g])[1]
    x1=g1.iloc[:,1:3]
    n1 = len(x1)

    g2=dict([i for i in data_g])[2]
    x2=g2.iloc[:,1:3]
    n2 = len(x2)

    g3 = dict([i for i in data_g])[3]
    x3 = g3.iloc[:, 1:3]
    n3=len(x3)

    p1=n1/(n1+n2+n3)
    p2=n2/(n1+n2+n3)
    p3=n3/(n1+n2+n3)

    u1 = x1.mean()
    u2 = x2.mean()
    u3 = x3.mean()

    S1 = np.mat(x1.cov())
    S2 = np.mat(x2.cov())
    S3 = np.mat(x3.cov())

    S1_ = np.linalg.inv(S1)
    S2_ = np.linalg.inv(S2)
    S3_ = np.linalg.inv(S3)

    b1 = math.log(np.linalg.det(S1))-2*math.log(p1)
    b2 = math.log(np.linalg.det(S2))-2*math.log(p2)
    b3 =math.log(np.linalg.det(S3))-2 * math.log(p3)

    ln = len(x)
    x=np.array(x)
    xu1 = x - x
    xu2 = x - x
    xu3 = x - x
    print(ln, x, u1, u2, u3)
    for i in range(ln):
        xu1[i] = x[i] - u1
        xu2[i] = x[i] - u2
        xu3[i] = x[i] - u3

    a1 = np.diagonal(np.dot(np.dot(xu1, S1_), np.transpose(xu1)))
    a2 = np.diagonal(np.dot(np.dot(xu2, S2_), np.transpose(xu2)))
    a3 = np.diagonal(np.dot(np.dot(xu3, S3_), np.transpose(xu3)))

    d1 = a1 + b1
    d2 = a2 + b2
    d3 = a3 + b3
    for i in range(ln):
        d1[i] = math.exp(-d1[i] / 2)
        d2[i] = math.exp(-d2[i] / 2)
        d3[i] = math.exp(-d3[i] / 2)

    P1 = d1/(d1+d2+d3)
    P2 = d2/(d1+d2+d3)
    P3 = d3 / (d1 + d2 + d3)

    result = np.zeros((3,3))
    for i in range(ln):
        a=P1[i]
        b=P2[i]
        c=P3[i]
        maxnum, lb = find_maxnum(a, b,c)
        at = label[i] - 1
        lb = lb-1
        result[at, lb] = result[at, lb]+1
    s = np.sum(np.sum(result))
    e = np.sum(np.diagonal(result))/s  #错误率
    e = 1-e

    #测试样本的类别
    sx = np.array(tx.iloc[:, 1:3])
    ln = len(sx)
    r = np.zeros(ln)

    xu1=sx-sx
    xu2 =sx - sx
    xu3 = sx - sx
    for i in range(ln):
        xu1[i]=sx[i]-u1
        xu2[i]=sx[i]-u2
        xu3[i] = sx[i] - u3

    a1=np.diagonal(np.dot(np.dot(xu1,S1_),np.transpose(xu1)))
    a2 =np.diagonal(np.dot(np.dot(xu2, S2_), np.transpose(xu2)))
    a3 = np.diagonal(np.dot(np.dot(xu3, S3_), np.transpose(xu3)))

    d1=a1+b1;
    d2=a2+b2;
    d3 = a3 + b3;
    for i in range(ln):
        d1[i]=math.exp(-d1[i]/2)
        d2[i] =math.exp(-d2[i]/2)
        d3[i] = math.exp(-d3[i] / 2)

    P1=d1/(d1+d2+d3)
    P2=d2/(d1+d2+d3)
    P3 = d3 / (d1 + d2+d3)

    for i in range(ln):
        a = P1[i]
        b = P2[i]
        c = P3[i]
        maxnum, lb = find_maxnum(a, b,c)
        r[i] = lb
    return r,e

r,e=Bayes_method_of_two_populations(g,tx)
print('申请者应属类别：',r)
print('分类方法的错误率：',e)

#交叉确认估计法
print('交叉确认估计法')
#针对G
n=len(g)
l=np.ones(n)
error=0
print('错误位置：')
for i in range(n):
    tx=g.iloc[[i]]
    g1=g.drop([i])

    r,e = Bayes_method_of_two_populations(g1,tx)
    l[i]=r[0]
    if l[i]!=lb[i]:
        print(i+1)
        error=error+1
e=error/n
print('分类方法的错误率：',e)
