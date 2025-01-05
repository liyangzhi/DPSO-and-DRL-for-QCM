def coup(nam):#输入的是结构限制图，输出的图的信息及边的列表
    coup = open('./couplings/'+nam, 'r')
    line1=coup.readline()#读取行
    a = list(map(int,line1.split()))#边：迭代器，转换成列表
    b=[(a[0],a[1])]
    line=coup.readline()
    while line!='\n':
        a = list(map(int,line.split()))
        b.append((a[0],a[1]))
        line=coup.readline()
    coup.close()
    print('限制图的边：',b)
    #assert 1==2
    return b