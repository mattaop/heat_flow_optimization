import random as r
import matplotlib.pyplot as plt

def printlist(v):
    for i in range(len(v)):
        print(v[i])

def f(x):
    return (x[0]-4)**2+(x[1]-4)**2

x=[]
for i in range(3):
    x.append([r.uniform(-1,1),r.uniform(-1,1)])
    x[i].append(f(x[i])) # Koordinater til ovn test
def sort(x):
    k=0
    while k!=len(x)-1:
        if x[k][2]>x[k+1][2]:
                a=x[k]
                x[k]=x[k+1]
                x[k+1]=a
                k=0
                continue
        k+=1
    return x
def centroid(x):
    return [0.5*(x[0][0]+x[1][0]),0.5*(x[0][1]+x[1][1])]
def direction(x,t):
    return [centroid(x)[0]+t*(x[2][0]-centroid(x)[0]),centroid(x)[0]+t*(x[2][0]-centroid(x)[0])]

sort(x)
k=0
while (x[1][2]-x[0][2])>0.0001:
    printlist(x)
    sort(x)
    k+=1
    print('hei')
    printlist(x)
    x_1=direction(x,-1)
    x_1f=f(x_1) # Koordinater til ovn test
    if x[0][2]<=x_1f<=x[1][2]:
        x_1.append(x_1f)
        x[2]=x_1
        print('1')
        continue
    elif x_1f<x[0][2]:
        x_2=direction(x,-2)
        x_2f=f(x_2) # Koordinater til ovn test
        if x_2f<x_1f:
            x_2.append(x_2f)
            x[2]=x_2
            print('2')
            continue
        else:
            x_1.append(x_1f)
            x[2]=x_1
            print('3')
            continue
    elif x_1f>=x[1][2]:
        if x[1][2]<=x_1f<x[2][2]:
            x_m12=direction(x,-0.5)
            x_m12f=f(x_m12) # Koordinater til ovn test
            if x_m12f<=x_1f:
                x_m12.append(x_m12f)
                x[2]=x_m12
                print('4')
                continue
        else:
            x_p12=direction(x,0.5)
            x_p12f=f(x_p12) # Koordinater til ovn test
            if x_p12f<x[2][2]:
                x_p12.append(x_p12f)
                x[2]=x_p12
                print('5')
                continue
        x[1][0]=0.5*(x[0][0]+x[1][0])
        x[1][1]=0.5*(x[0][1]+x[1][1])
        x[1][2]=f([x[1][0],x[1][1]]) # Koordinater til ovn test
        x[2][0]=0.5*(x[0][0]+x[2][0])
        x[2][1]=0.5*(x[0][1]+x[2][1])
        x[2][2]=f([x[2][0],x[2][1]]) # Koordinater til ovn test
        continue
print(x[0])
print(x[0][2])
print(x[1][2]-x[0][2])
print('Iterations',k)
