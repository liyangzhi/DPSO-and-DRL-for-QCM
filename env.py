from gym.spaces import Discrete,Box
import numpy as np
import random
import copy

#还是没有继承，看是否需要吧
class Environment:
    def __init__(self,topology,cir):
        
        self.ini_mp=cir[0]#初始映射
        self.ini_s=cir[1]#初始状态
        self.mp=copy.deepcopy(self.ini_mp)#深度复制
        self.lc=copy.deepcopy(self.ini_s)
        self.pc=[]
       
        self.nq=len(self.mp)#逻辑比特数
        self.ng=len(self.ini_s)#双量子门数
        self.pq=topology[0][0]
        #拓扑结构图的初始化，一般用于限制动作空间
        self.topology=topology[1:]#可能还需要根据映射而具体化，选出拓扑结构的子图，这样动作会更加明确
        
        #动作空间与状态空间
        # self.action_space = Discrete(len(self.possible_actions))#根据拓扑结构图的边确定有多少种动作
        self.possible_actions = self.__get_possible_actions()#确定当前状态可执行的动作，也就是边,这里应该是所有可执行的动作吧，不能是单个门可执行的动作吧
        #self.action_sapce=Discrete(4)#网格动作：上下左右，或者根据拓扑图决定动作空间，动作空间为离散，是为了随机取样只需要完成相应动作的序号对应 
        self.action_space=Discrete(len(self.possible_actions))
        #需进一步优化的地方：变量范围
        self.observation_space=Box(low = -np.inf, high = np.inf, shape=(self.pq,self.ng,), dtype=np.int64)#它的范围变化应该在0-量子比特数量之间
        #oce = spaces.Box(-np.inf, np.inf, shape=(2,2,), dtype=np.int64)
        #若是有问题尝试
        #self.observation_space=Box(low = -np.inf, high = np.inf, shape=(1,self.ng,2,), dtype=np.int64)
        
        self.state=None#声明状态
        
        
    #用于复原环境的初始值，关注那些在step函数中变化的量
    def reset(self):
        reob=copy.deepcopy(self.ini_s)
        self.mp=copy.deepcopy(self.ini_mp)
        self.lc=copy.deepcopy(self.ini_s)
        self.pc=[]
        self.state=np.full(shape=(self.pq,self.ng),fill_value=-2)
        #self.state=np.zeros((self.pq,self.ng))
        #print(self.pq,self.ng)
        for i in range(self.ng):
            k1=self.mp[reob[i][0]]
            k2=self.mp[reob[i][1]]
            self.state[k1][i]=reob[i][0]
            self.state[k2][i]=reob[i][1]
    
        return self.state

    
    #奖励函数在这里使用
    def step(self,action):
        (op1,op2)=self.possible_actions[action]#获取跳转的行
        tmp=copy.deepcopy(self.state[op1][:])
        self.state[op1][:]=self.state[op2][:]
        self.state[op2][:]=tmp
        self.pc.append([op1,op2])
        reward=0
        done=False
        info=[]
        
        #为了后面的状态特征，必须更新当前量子线路lc，以及映射mp，为了计算深度可能还必须使用实际线路pc
        mpv=list(self.mp.values())#逻辑量子比特的映射值
        q1_index=mpv.index(op1)
        q2_index=mpv.index(op2)
        self.mp[q1_index]=op2
        self.mp[q2_index]=op1
        
        while True:
            cl=self.state[:,0]
            act=[]
            for i in range(self.pq):
                if cl[i]!=-2:
                    act.append(i)
            rva=copy.deepcopy(act)
            rva.reverse()
            if (act in self.possible_actions) or (rva in self.possible_actions):#
                self.lc.pop(0)
                self.pc.append(act)
                self.state[:,:-1]=self.state[:,1:]
                self.state[:,-1]=[-2]*self.pq
                reward+=1#使得当前门可执行的奖励
                if (self.state==-2).all():
                    reward+=10
                    break
            else:
                if reward==0:
                    reward-=1
                break
        
        if (self.state==-2).all():
            done=True
        reward-=30#这里为何减30呢？
        
        return self.state,reward,done,info
    
    #根据映射，这里是当前映射所有可执行的动作
    def __get_possible_actions(self):#就是限制图的边
        pa=[]
        ex_q=list(self.ini_mp.values())
        #print(ex_q)
        for (i,j) in self.topology:
            if (i in ex_q) and (j in ex_q):
                pa.append([i,j])       
        return pa

                
        
    #用于根据当前状态获取可执行的动作
    def get_act(self):
        acts=[]
        CG=nx.Graph()
        #不能这样，必须从映射的结点及边连接建立图
        CG.add_nodes_from(list(self.ini_mp.values()))
        CG.add_edges_from(self.possible_actions)
        #找到两条边
        while True:
            cl=self.state[:,0]
            #print(cl)
            act=[]
            for i in range(self.pq):
                if cl[i]!=-2:
                    act.append(i) #获取当前门，下一步需要对门的连接进行判断
            
            if len(act)<2:
                print(act)
                print(cl)
                assert 1==2
            arv=copy.deepcopy(act)
            arv.reverse()

            if (act in self.possible_actions) or (arv in self.possible_actions):#
                self.state[:,:-1]=self.state[:,1:]
                self.state[:,-1]=[-2]*self.pq
                self.lc.pop(0)
                self.pc.append(act)
                
            else:#貌似act为空也可以跳出去
                break
            
        if len(act)==0:
            print('已完成最近邻')
            assert 1==2
        
        if len(act)<2:
            print(act)
            assert 1==2
         
        #print(act)
        lujing=[p for p in nx.all_shortest_paths(CG,source=act[0],target=act[1])]
        #print(lujing)

        for r in range(len(self.possible_actions)):
            if (act[0] in self.possible_actions[r]) or (act[1] in self.possible_actions[r]):
                for p in lujing:
                    #print(p)
                    if ( self.possible_actions[r][0] in p) and (self.possible_actions[r][1] in p):
                        acts.append(r)
                    
        if len(acts)==0:
            print(lujing)
            print(act)
            assert 1==2
                        
        
        return acts
        
        
    def depth(self,num_q,c_gates):
        d = [0] * num_q

        for [q1,q2] in c_gates:
            d_max = max(d[q1], d[q2])
            d[q1] = d_max + 1
            d[q2] = d_max + 1

        return max(d)

    def get_obs(self):#需要关心当前量子线路与映射
        qn=0#量子比特数
        sd=0#深度
        cxg=0#cnot门
        cp=0#通信比
        cd=0#关键深度
        pr=0#并行度
        lvr=0#生命度
        md=0#平均最短路径
        mxd=0#最大度
        mid=0#最小度
        bzc=0#标准差
        #3种普适特征
        qn=self.nq
        sd=depth(qn,self.lc)#有点问题，可能还需要实时更新量子线路中的逻辑门
        #print(sd)
        cxg=len(self.lc)
        if sd==0:
            cp=0#通信比
            cd=0#关键深度
            pr=0#并行度
            lvr=0#生命度
            md=0#平均最短路径
            mxd=0#最大度
            mid=0#最小度
            bzc=0#标准差
        else: 
            #4种复杂特征
            jiedian=self.ini_mp.keys()
            DG=nx.Graph()
            DG.add_nodes_from(jiedian)
            DG.add_edges_from(self.lc)
            #通信度
            all_dg=dict(DG.degree())
            du=list(all_dg.values())
            sud=sum(du)
            cp=sud/(qn*(qn-1))
            #关键深度
            cd=sd/cxg
            #并行度：
            pr=(cxg/sd-1)/(qn-1)
            #生命度
            # 创建一个nxd全零矩阵
            #print(sd)
            matrix = np.zeros((qn, sd))
            # 随机初始化矩阵的每个元素
            ind=0
            #print('nihao')
#             for (i,j) in self.lc:
#                 print(i,j)
#                 if (matrix[i, ind]==0) and (matrix[j, ind]==0):
#                     matrix[i, ind]=1
#                     matrix[j, ind]=1
#                 else:
#                     ind+=1
#                     #print(ind)
#                     matrix[i, ind]=1
#                     matrix[j, ind]=1

            #lvr=np.sum(matrix)/(qn*sd)#感觉就是门的比特数乘以2就ok了因为没有单量子门
            lvr=(2*cxg)/(qn*sd)
            #4种交互特征
            CG=nx.Graph()
            CG.add_nodes_from([1, 2, 3, 4, 5,7,8,9,10,11,12,13,14,15,16,17,18,19])
            CG.add_edges_from(self.topology)
            #计算平均最短路径
            tcs=[tuple(x) for x in self.lc]
            countg=Counter(tcs)
            md=0
            for k in range(cxg):
                #应该是否要使用初始量子线路作为标准
                dg=self.lc[k]#当前双量子门
                x=self.mp[dg[0]]#第一量子比特对应的物理比特
                y=self.mp[dg[1]]#第二个量子比特对应的物理量子比特
                md+=(nx.shortest_path_length(CG,x,y)-1)/(countg[(dg[0],dg[1])]+countg[(dg[1],dg[0])])
            #最大、最小结点的度
            mxd=max(du)
            mid=min(du)
            #边权分布标准差
            wb=list(countg.keys())
            WG=nx.Graph()
            WG.add_nodes_from([i for i in range(qn)])
            for i in range(len(wb)):
                w=countg[wb[i]]+countg[(wb[i][1],wb[i][0])]
                WG.add_edge(wb[i][0],wb[i][1],weight=w)
            bzc=np.std(nx.adjacency_matrix(WG).A)
        
        
        
#         mpv=list(self.mp.values())
#         obs=np.zeros((20,20))
#         for (i,j) in self.lc:
#             p1=mpv[i]
#             p2=mpv[j]
#             obs[p1][p2]+=1
            
        return [qn,sd,cxg,cp,cd,pr,lvr,md,mxd,mid,bzc]
            
    #用于画图显示，为了直观，这里可以先不考虑
    def render(self):
        pass
    