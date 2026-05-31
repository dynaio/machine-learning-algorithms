import math, random
lamda = 0.5
mu = 1.0
n = 100
tin = 0
tout = -1/mu*math.log(random.random())

Tq   = [] 
Tin  = []
Ws   = []
fsum = []
Tq.append(0)

for i in range(n-5):
    Tin.append(-1/lamda*math.log(random.random())) 
    #Ws.append(-1/mu*math.log(random.random())) 
    #print(f"C{i+1} : Tin ={Tin [i]:.2f}, temps de service ={Ws[i]:.2f}")
    Tin.sort()

for i in range(n):
    fsum.append(lamda * math.exp(-lamda*Tin[i]))
print (fsum)
for i in range(n):
    cond = Tq[i] + Ws[i]- Tin[i]
    if (cond)>0:  
        Tq.append(cond) 
    else:
        Tq.append(0)
    print(f"Temp d'attdente = {Tq[i]}")    



