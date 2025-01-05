import qiskit
import random
import numpy as np
from pathlib import Path
import copy

def cir2num(nam):#输入的是电路名
    cirt = open('./circuits/'+nam, 'r')
    a=""
    line=cirt.readline()#读取一行
    a = list(line.split())
    #print(a)
    #assert 1==2
    while a[0]!="qreg":
        line=cirt.readline()
        a = list(line.split())
        #print(a)
        #assert 1==2
        if len(a)==0:#跳过空行
            line=cirt.readline()
            a = list(line.split())
            print('我执行过！')
            pass
    #assert 1==2
    #因为一行一行的读取，因此是为了跳过前两行
    b=[int(a[1][2:len(a[1])-2])]#获取量子比特
    #print(a)
    print(b)
    #assert 1==2
    #assert 1==2
    line=cirt.readline()#第4行，经典寄存器
    while True:
        if not line:#当qasm读取完毕，跳出循环
            break
        a = list(line.split())#量子门
        if len(a)==0:#跳过空行
            line=cirt.readline()
            a = list(line.split())
            pass
        if a[0]=='creg':#判断经典寄存器，读取下一行
            line=cirt.readline()
            a = list(line.split())

        if len(a)==3:#判断多量子门
            #print(a)
            c=a[1][0:len(a[1])-1].split(',')
            #print(c)
            e=int(c[0][2:len(c[0])-1])
            #print(e)
            f=int(a[2][2:len(a[2])-2])
            #print(f)
            #assert 1==2
            b.append((e,f))
        else:#判断单、双量子门
            #print(a)
            c=a[1][0:len(a[1])-1].split(',')
            #print(c)
            #assert 1==2
            if len(c)==2:#判断双量子门
                #print("双量子门哟：")
                #print(c)
                #c=a[1][0:len(a[1])-1].split(',')
                #print(c)
                #assert 1==2
                e=int(c[0][2:len(c[0])-1])
                f=int(c[1][2:len(c[1])-1])
                #print(e)
                #print(f)
                #assert 1==2
                b.append((e,f))
                pass
        line=cirt.readline()
    cirt.close()
    print('量子比特数及所有双量子门',b)
    print("双量子门数量：",len(b)-1)
    return  b