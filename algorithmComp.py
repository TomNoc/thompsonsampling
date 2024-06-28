import math
import statistics
import matplotlib.pyplot as plt
import numpy as np
from random import *
from scipy.stats import bernoulli 
from scipy.stats import beta

class Arm:
    mean = 0
    total = 0
    pulled = 0

    def __init__(self, mean):
        self.mean = mean

    def getAverage(self):
        if(self.pulled == 0):
            return self.total
        else:
            return self.total/self.pulled
    
    def pull(self):
        value = bernoulli.rvs(self.mean)
        self.total += value
        self.pulled += 1
        return value

def getOptimal(arms, k, t):
    bestMean = 0
    for i in range(0,k):
        if(arms[i].mean > bestMean):
            bestMean = arms[i].mean
    return bestMean * t

def exploit(arms, k):
    best = 0
    bestAvg = 0
    currAvg = 0
    for i in range(0,k):
        currAvg = arms[i].getAverage()
        if(currAvg > bestAvg):
            best = i
            bestAvg = currAvg
    return best

def softmax(samples, learn): #Lower learn value more uniform
    prob = []
    wtavg = 0
    rand = random()
    
    for j in range(0,len(samples)):
        wtavg += math.exp(learn*samples[j])
    for i in range(0,len(samples)):
        prob.append(math.exp(learn*samples[i])/wtavg)
    for k in range(0,len(prob)):
        if rand<prob[k]:
            return [k, prob]
        else:
            rand = rand - prob[k]

k = 3
armMeans = []
# armMeans = [0.5,0.55,0.6]

def generateMeans():
    for i in range(0,k):
        armMeans.append(random())

def ProbThompsonSample():
    #Configure
    rounds = 1000

    arms = []
    armHistory = []
    reward = 0
    optimal = max(armMeans)
    pseudoRegret = 0
    pseudoRegrets = []

    a = [] #beta distribution values
    b = []

    #load array of arm objects with random mean value
    for i in range(0,k):
        newArm = Arm(mean=armMeans[i]) #mean=random()
        arms.append(newArm)

    #run probabilistic Thompson Sampling
    for j in range(0,k):
            a.append(1)
            b.append(1)
    for i in range(0,rounds):
        sample = []
        for j in range(0,k):
            sample.append(beta.rvs(a[j], b[j]))

        #pull arm
        chosenArm = softmax(sample,100)[0] #change learning rate for different results
        currReward = arms[chosenArm].pull()
        reward += currReward
        armHistory.append(chosenArm)
        pseudoRegret += optimal - armMeans[chosenArm]
        pseudoRegrets.append(pseudoRegret)

        #update distribution
        a[chosenArm] += currReward
        b[chosenArm] += 1 - currReward

    return [reward, optimal*rounds, pseudoRegrets]

def ProbThompsonSample2():
    #Configure
    rounds = 1000

    arms = []
    armHistory = []
    reward = 0
    optimal = max(armMeans)
    pseudoRegret = 0
    pseudoRegrets = []
    sampleHistory = []

    a = [] #beta distribution values
    b = []

    #load array of arm objects with random mean value
    for i in range(0,k):
        newArm = Arm(mean=armMeans[i]) #mean=random()
        arms.append(newArm)

    #run probabilistic Thompson Sampling
    for j in range(0,k):
            a.append(1)
            b.append(1)
    for i in range(0,rounds):
        currsample = []
        for j in range(0,k):
            currsample.append(beta.rvs(a[j], b[j]))
        sampleHistory.append(currsample)
        sample = []

        for j in range(0,k):
            sum = 0
            for r in range(0,len(sampleHistory)):
                sum += sampleHistory[r][j]
            sample.append(sum)

        #pull arm
        chosenArm = softmax(sample,0.5)[0] #change learning rate for different results
        currReward = arms[chosenArm].pull()
        reward += currReward
        armHistory.append(chosenArm)
        pseudoRegret += optimal - armMeans[chosenArm]
        pseudoRegrets.append(pseudoRegret)

        #update distribution
        a[chosenArm] += currReward
        b[chosenArm] += 1 - currReward

    return [reward, optimal*rounds, pseudoRegrets]


def ThompsonSample():
    #Configure
    rounds = 1000

    arms = []
    armHistory = []
    reward = 0
    optimal = max(armMeans)
    pseudoRegret = 0
    pseudoRegrets = []

    a = [] #beta distribution values
    b = []

    #load array of arm objects with random mean value
    for i in range(0,k):
        newArm = Arm(mean=armMeans[i]) #mean=random()
        arms.append(newArm)

    #run probabilistic Thompson Sampling
    for j in range(0,k):
            a.append(1)
            b.append(1)
    for i in range(0,rounds):
        sample = []
        for j in range(0,k):
            sample.append(beta.rvs(a[j], b[j]))

        #pull arm
        chosenArm = np.argmax(sample)
        currReward = arms[chosenArm].pull()
        reward += currReward
        armHistory.append(chosenArm)
        pseudoRegret += optimal - armMeans[chosenArm]
        pseudoRegrets.append(pseudoRegret)

        #update distribution
        a[chosenArm] += currReward
        b[chosenArm] += 1 - currReward

    return [reward, optimal*rounds, pseudoRegrets]

def Greedy():
    #Configure
    rounds = 1000

    arms = []
    armHistory = []
    reward = 0
    optimal = max(armMeans)
    pseudoRegret = 0
    pseudoRegrets = []

    #load array of arm objects with random mean value
    for i in range(0,k):
        newArm = Arm(mean=armMeans[i]) #mean=random()
        arms.append(newArm)

    for i in range(1,rounds+1):
        prob = math.pow(i,-1/3) * math.pow(k*math.log(i),1/3)
        if(random() <= prob):
            #explore
            curr = randint(0,k-1)
            reward += arms[curr].pull()
            armHistory.append(curr)
            pseudoRegret += optimal - armMeans[curr]
            pseudoRegrets.append(pseudoRegret)
        else:
            #exploit
            curr = exploit(arms, k)
            reward += arms[curr].pull()
            armHistory.append(curr)
            pseudoRegret += optimal - armMeans[curr]
            pseudoRegrets.append(pseudoRegret)
    return [reward, optimal*rounds, pseudoRegrets]

def Exp3():
    #Configure
    rounds = 1000

    arms = []
    armHistory = []
    reward = 0
    optimal = max(armMeans)
    pseudoRegret = 0
    pseudoRegrets = []

    #load array of arm objects with random mean value
    for i in range(0,k):
        newArm = Arm(mean=armMeans[i]) #mean=random()
        arms.append(newArm)

    St = []
    curr = 0

    for i in range(0,k):
        St.append(0)
    
    for i in range(0,rounds):
        curr = softmax(St,0.1)
        r = arms[curr[0]].pull()
        for j in range(0,k):
            if (j == curr[0]):
                St[j] = St[j] + 1 - (1 - r)/curr[1][j]
            else:
                St[j] = St[j] + 1
        
        reward += r
        armHistory.append(curr[0])
        pseudoRegret += optimal - armMeans[curr[0]]
        pseudoRegrets.append(pseudoRegret)
    return [reward, optimal*rounds, pseudoRegrets]

#Regret on 1000 rounds
generateMeans()
y1 = ProbThompsonSample()[2]
y2 = ThompsonSample()[2]
y3 = Greedy()[2]
y4 = Exp3()[2]
y5 = ProbThompsonSample2()[2]

sim = [y1,y2,y3,y4,y5]

maxy = max([sim[0][len(sim[0])-1],sim[1][len(sim[1])-1],sim[2][len(sim[2])-1],sim[3][len(sim[3])-1],sim[4][len(sim[4])-1]])

plt.plot(sim[0])
plt.plot(sim[1])
plt.plot(sim[2])
plt.plot(sim[3])
plt.plot(sim[4])
plt.suptitle('Regret on 1000 Rounds')
plt.legend(["Probabilistic TS","Deterministic TS","Greedy","Exp3","Probabilistic TS2"], loc="upper left")
plt.xlim(0, 1000)
plt.ylim(0, maxy+10)
plt.xlabel("Round")
plt.ylabel("Pseudoregret")
plt.show()

#Average Regret on 100 simulations
simRuns=100
histories1 = []
histories2 = []
histories3 = []
histories4 = []
histories5 = []
expectedRegret1 = []
expectedRegret2 = []
expectedRegret3 = []
expectedRegret4 = []
expectedRegret5 = []

for i in range(0,simRuns):
    generateMeans()
    histories1.append(ProbThompsonSample()[2])
    histories2.append(ThompsonSample()[2])
    histories3.append(Greedy()[2])
    histories4.append(Exp3()[2])
    histories5.append(ProbThompsonSample2()[2])
for j in range(0,len(histories1[1])):
    sumRegret1 = 0
    sumRegret2 = 0
    sumRegret3 = 0
    sumRegret4 = 0
    sumRegret5 = 0
    for i in range(0,len(histories1)):
        sumRegret1 += histories1[i][j]
        sumRegret2 += histories2[i][j]
        sumRegret3 += histories3[i][j]
        sumRegret4 += histories4[i][j]
        sumRegret5 += histories5[i][j]
    expectedRegret1.append(sumRegret1/len(histories1))
    expectedRegret2.append(sumRegret2/len(histories2))
    expectedRegret3.append(sumRegret3/len(histories3))
    expectedRegret4.append(sumRegret4/len(histories4))
    expectedRegret5.append(sumRegret5/len(histories5))

ymax2 = max([expectedRegret1[len(expectedRegret1)-1],expectedRegret2[len(expectedRegret2)-1],expectedRegret3[len(expectedRegret3)-1],expectedRegret4[len(expectedRegret4)-1],expectedRegret5[len(expectedRegret5)-1]])

plt.plot(expectedRegret1)
plt.plot(expectedRegret2)
plt.plot(expectedRegret3)
plt.plot(expectedRegret4)
plt.plot(expectedRegret5)
plt.suptitle('Regret on 1000 Simulations')
plt.legend(["Probabilistic TS","Deterministic TS","Greedy","Exp3","Probabilistic TS2"], loc="upper left")
plt.xlim(0, 1000)
plt.ylim(0, ymax2+10)
plt.xlabel("Round")
plt.ylabel("Expected Regret")
plt.show()

# #Average Regret on Worst 10% Simulations
lastRegret1 = []
lastRegret2 = []
lastRegret3 = []
lastRegret4 = []
lastRegret5 = []
worstRegret1 = []
worstRegret2 = []
worstRegret3 = []
worstRegret4 = []
worstRegret5 = []
for i in range(0,len(histories1)):
    lastRegret1.append(histories1[i][len(histories1[i])-1])
    lastRegret2.append(histories2[i][len(histories2[i])-1])
    lastRegret3.append(histories3[i][len(histories3[i])-1])
    lastRegret4.append(histories4[i][len(histories4[i])-1])
    lastRegret5.append(histories5[i][len(histories5[i])-1])

lastRegret1.sort(reverse=True)
lastRegret2.sort(reverse=True)
lastRegret3.sort(reverse=True)
lastRegret4.sort(reverse=True)
lastRegret5.sort(reverse=True)

worstRegret1 = lastRegret1[:100]
worstRegret2 = lastRegret2[:100]
worstRegret3 = lastRegret3[:100]
worstRegret4 = lastRegret4[:100]
worstRegret5 = lastRegret5[:100]

worstRegretH1 = []
worstRegretH2 = []
worstRegretH3 = []
worstRegretH4 = []
worstRegretH5 = []

for i in range(0,len(histories1)):
    for j in range(0,len(worstRegret1)):
        if (worstRegret1[j] == histories1[i][len(histories1[i])-1]):
            worstRegretH1.append(histories1[i])

for i in range(0,len(histories2)):
    for j in range(0,len(worstRegret2)):
        if (worstRegret2[j] == histories2[i][len(histories2[i])-1]):
            worstRegretH2.append(histories2[i])

for i in range(0,len(histories3)):
    for j in range(0,len(worstRegret3)):
        if (worstRegret3[j] == histories3[i][len(histories3[i])-1]):
            worstRegretH3.append(histories3[i])

for i in range(0,len(histories4)):
    for j in range(0,len(worstRegret4)):
        if (worstRegret4[j] == histories4[i][len(histories4[i])-1]):
            worstRegretH4.append(histories4[i])

for i in range(0,len(histories5)):
    for j in range(0,len(worstRegret5)):
        if (worstRegret5[j] == histories5[i][len(histories5[i])-1]):
            worstRegretH5.append(histories5[i])

expectedWorstRegret1 = []
expectedWorstRegret2 = []
expectedWorstRegret3 = []
expectedWorstRegret4 = []
expectedWorstRegret5 = []

for j in range(0,len(worstRegretH1[1])):
    sumRegret1 = 0
    for i in range(0,len(worstRegretH1)):
        sumRegret1 += worstRegretH1[i][j]
    expectedWorstRegret1.append(sumRegret1/len(worstRegretH1))

for j in range(0,len(worstRegretH2[1])):
    sumRegret2 = 0
    for i in range(0,len(worstRegretH2)):
        sumRegret2 += worstRegretH2[i][j]
    expectedWorstRegret2.append(sumRegret2/len(worstRegretH2))

for j in range(0,len(worstRegretH3[1])):
    sumRegret3 = 0
    for i in range(0,len(worstRegretH3)):
        sumRegret3 += worstRegretH3[i][j]
    expectedWorstRegret3.append(sumRegret3/len(worstRegretH3))

for j in range(0,len(worstRegretH4[1])):
    sumRegret4 = 0
    for i in range(0,len(worstRegretH4)):
        sumRegret4 += worstRegretH4[i][j]
    expectedWorstRegret4.append(sumRegret4/len(worstRegretH4))

for j in range(0,len(worstRegretH5[1])):
    sumRegret5 = 0
    for i in range(0,len(worstRegretH5)):
        sumRegret5 += worstRegretH5[i][j]
    expectedWorstRegret5.append(sumRegret5/len(worstRegretH5))

ymax3 = max([expectedWorstRegret1[len(expectedWorstRegret1)-1],expectedWorstRegret2[len(expectedWorstRegret2)-1],expectedWorstRegret3[len(expectedWorstRegret3)-1],expectedWorstRegret4[len(expectedWorstRegret4)-1],expectedWorstRegret5[len(expectedWorstRegret5)-1]])

plt.plot(expectedWorstRegret1)
plt.plot(expectedWorstRegret2)
plt.plot(expectedWorstRegret3)
plt.plot(expectedWorstRegret4)
plt.plot(expectedWorstRegret5)
plt.suptitle('Expected Regret on Worst 10% Simulations')
plt.legend(["Probabilistic TS","Deterministic TS","Greedy","Exp3","Probabilistic TS2"], loc="upper left")
plt.xlim(0, 1000)
plt.ylim(0, ymax3+10)
plt.xlabel("Round")
plt.ylabel("Expected Regret")
plt.show()
