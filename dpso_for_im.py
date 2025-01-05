from two_qubit_gate import cir2num
from coupling_graph import coup
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
import copy

cp='tokyo.txt'
cr='alu-v3_35.qasm'
cir=cir2num(cr)#输出的是逻辑比特以及双量子门
arch=coup(cp)#输出的是物理量子比特、边数，以及连接边
print(cir[0])
print(len(cir)-1)
#输入的量子线路的参数提取
lq=cir[0]#逻辑量子比特数目
pq=arch[0][0]#逻辑量子比特数目
print(lq)
print(pq)
num_g=len(cir)


class DPSO():
	# PSO参数设置
    def __init__(self,pN, max_iter, lq, pq, cir, num_g,arch,c1=0.1,c2=0.2,c3=0.3):
        self.c1 = 0.1
        self.c2 = 0.2
        self.c3=c3
        self.pN = pN  # 粒子种群数量
        self.max_iter = max_iter  # 迭代次数
        self.lq=lq#逻辑量子比特数量，粒子维度
        self.pq=pq#物理量子比特数量
        self.cir=cir#原电路
        self.arch=arch#结构
        self.num_g=num_g#量子门数
        
        self.X = np.zeros((self.pN, self.lq),dtype=int)  # 所有粒子的位置,映射
        self.pbest = np.zeros((self.pN, self.lq),dtype=int)  # 个体经历的最佳位置
        self.gbest = np.zeros((1, self.lq),dtype=int)#全局最佳位置
        self.p_fit = np.zeros(self.pN,dtype=int)  # 每个个体的历史最佳适应值
        self.fit = 1e10  # 全局最佳适应值
        self.countg=Counter(cir)
 
    #目标函数——适应度函数
    def fitness(self,num_g,imp):
        nnc=0
        zuobiao={}
        zuobiao[0]=[0,0]
        zuobiao[1]=[1,0]
        zuobiao[2]=[2,0]
        zuobiao[3]=[3,0]
        zuobiao[4]=[4,0]
        zuobiao[5]=[0,-1]
        zuobiao[6]=[1,-1]
        zuobiao[7]=[2,-1]
        zuobiao[8]=[3,-1]
        zuobiao[9]=[4,-1]
        zuobiao[10]=[0,-2]
        zuobiao[11]=[1,-2]
        zuobiao[12]=[2,-2]
        zuobiao[13]=[3,-2]
        zuobiao[14]=[4,-2]
        zuobiao[15]=[0,-3]
        zuobiao[16]=[1,-3]
        zuobiao[17]=[2,-3]
        zuobiao[18]=[3,-3]
        zuobiao[19]=[4,-3]
        
        
        duijiao=[(1,7),(2,6),(3,9),(4,8),(5,11),(6,10),(7,13),(8,12),(11,17),(12,16),(13,19),(14,18)]
        
        
        G=nx.Graph()
        G.add_nodes_from([1, 2, 3, 4, 5,7,8,9,10,11,12,13,14,15,16,17,18,19])
        G.add_edges_from(self.arch)
        
        
        
        for k in range(num_g):
            #应该是否要使用初始量子线路作为标准
            dg=self.cir[k]#当前双量子门
            x=imp[dg[0]]#第一量子比特对应的物理比特
            y=imp[dg[1]]#第二个量子比特对应的物理量子比特
            nnc+=(nx.shortest_path_length(G,x,y)-1)/(self.countg[(dg[0],dg[1])]+self.countg[(dg[1],dg[0])])#/(self.cir.count((dg[0],dg[1]))+self.cir.count((dg[1],dg[0])))
            
#             x_zb=zuobiao[x]#第一量子比特的坐标
#             y_zb=zuobiao[y]#第二个量子比特的坐标
#             #计算最近邻代价
#             if (x,y) in duijiao or (y,x) in duijiao:
#                 nnc+=0
#             else:
#                 nnc+=abs(x_zb[0]-y_zb[0])+abs(x_zb[1]-y_zb[1])-1
        return nnc
    
    #设置变异因子c3的值
    def set_c3(self, value):
        self.c3= value
 
    #初始化种群
    def init_Population(self):
        #random.seed(1)#设置随机种子，调试随机数的影响
        for i in range(self.pN):      #因为要随机生成pN个数据，所以需要循环pN次
            print(i)
            pq_p=[]#物理量子比特的索引
            for j in range(self.pq):
                pq_p.append(j)
            imp=[]#映射关系，索引对应逻辑量子比特序号，值代表对应的物理量子比特
            for k in range(self.lq):#随机生成一种映射
                a=random.choice(pq_p)
                imp.append(a)
                pq_p.remove(a)
                
            #print(imp)
            #assert 1==2

            self.X[i]=imp#调用信道分配函数
#             print(self.X[i])
#             print(self.X)
            
            self.pbest[i] = copy.deepcopy(self.X[i])     #其实就是给self.pbest定值
            #print(self.X[i])
            tmp = self.fitness(self.num_g,self.X[i])  #得到现在最优
            #print(self.X[i])
            self.p_fit[i] = tmp    #这个个体历史最佳的位置
            if tmp < self.fit:   #得到现在最优和历史最优比较大小，如果现在最优小于历史最优，则更新历史最优
                self.fit = tmp
                self.gbest = copy.deepcopy(self.X[i])#这里太多赋值了，注意一下，万一出问题了，可改为deepcopy
            #print(self.X[i])
            
        #print(self.X)
 
    # 更新粒子位置
    def iterator(self):
        myfitness = []#记录每次迭代适应度
        #print(self.c3)
        #random.seed(2)#设置随机种子，画图1的查看随机数的影响
        for t in range(self.max_iter):    #迭代次数，不是越多越好
            print('迭代次数',t)
            for i in range(self.pN):#根据DPSO位置更新公式更新信道分配
                #print(self.X[i])
                for j in range(self.lq):
                    #self.X[i][j]=
                    #第一阶段F1
                    r1=random.uniform(0, 1)#系统随机数1
                    if r1<=self.c1:
                        F1=self.pbest[i][j]
                    else:
                        F1=self.X[i][j]
                    #第二阶段F2
                    #r2=random.uniform(0, 1)#系统随机数2，第二阶段可重新生成
                    if r1<=self.c2:
                        F2=self.gbest[j]
                    else:
                        F2=F1
                    #第三阶段F3
                    #r3=random.uniform(0, 1)#系统随机数3，第三阶段可重新生成
                    while True:
                        x_new=random.randrange(0,pq)
                        if x_new != self.X[i][j]:
                            break
#                     print(x_new)
#                     assert 1==2
#                     print(list(self.X[i]).index(3))
#                     assert 1==2
                    if r1<=self.c3:
                        if x_new not in self.X[i]:
                            self.X[i][j]=x_new
                        else:
                            md=list(self.X[i]).index(x_new)
                            self.X[i][md]=self.X[i][j]
                            self.X[i][j]=x_new
                    else:
                        if (F2 not in self.X[i]) or (j==list(self.X[i]).index(F2)):
                            self.X[i][j]=F2
                        else:
                            wc=list(self.X[i]).index(F2)
                            self.X[i][wc]=self.X[i][j]
                            self.X[i][j]=F2
#                 print(self.X[i])
#                 assert 1==2
            
            for i in range(self.pN):  # 更新gbest\pbest
                temp = self.fitness(self.num_g,self.X[i])
                if temp < self.p_fit[i]:  # 更新个体最优
                    self.p_fit[i] = temp
                    self.pbest[i] = copy.deepcopy(self.X[i])
                    if self.p_fit[i] < self.fit:  # 更新全局最优
                        self.gbest = copy.deepcopy(self.X[i])
                        self.fit = copy.deepcopy(self.p_fit[i])
                
#                 #信道融合阶段：判断位置更新后是否满足接口约束
#                 n_k=maxi_channels(self.X[i])
#                 if n_k>R:
#                     self.X[i]=channels_as(self.N,self.R,self.K,self.dim)#调用信道分配函数
                        
            #chongtu=self.fit/dim
            
            myfitness.append(self.fit)#存储当前迭代适应度值
            #fitness.append(chongtu)
            #print(self.X[0], end=" ")
            #print(self.fit)  # 输出最优值
        return myfitness


pN=50#种群大小
max_iter=100#最大迭代次数
c1=0.1
c2=0.2
c3=0.3

    
my_pso = DPSO(pN, max_iter, lq, pq, cir, num_g,arch,c1,c2,c3)#初始化DPSO类
my_pso.init_Population()#初始化种群
print(my_pso.X)

my_pso.set_c3(0.1)#设置变异因子                                                                                                                                                                                                                                                                                                                                                                                                       
fn1 = my_pso.iterator()#迭代更新，获取适应度值

print(my_pso.X)
print(my_pso.gbest)

my_imap=my_pso.gbest

