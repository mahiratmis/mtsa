

"""#This version is same as earlier version _v3

# There are 4 possible variants of _vs
# Initial population: Restricted, Unrestricted
# Mutation: Switch only, Switch and Open new Cluster

####This version Unrestricted + Switch Only ######

# For all variants we use following GA parameters:
# Population: 100
# Generation:25
# Crossover:0.8
# Mutation:0.4
# Gene Mutation:0.4 """

import numpy as np
import math
import json
import time
import random
from deap import base
from deap import creator
from deap import tools
import sys
import os
import csv

import cProfile
import pstats
import StringIO
import operator

from KmedianResultsParser import get_n_best_individuals

import concurrent.futures as cf

sys.path.append(os.getcwd())

# import sframe as sf
# from bokeh.charts import BoxPlot, output_notebook, show
# import seaborn as sns
# import pulp
# from pulp import *

# Import DEAP from and documentation
# https://deap.readthedocs.io/en/master/
# pip install deap


# Below functions are needed for Queuing approximation
# Define the generator that will generate all possible assignments of classes
# to servers, without permutations.


def generateVectorsFixedSum(m, n):
    """ generator for all combinations of $w$ for given number of servers and
     classes """
    if m == 1:
        yield [n]
    else:
        for i in range(n + 1):
            for vect in generateVectorsFixedSum(m - 1, n - i):
                yield [i] + vect


class DecorateProfiler(object):
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retVal = self.fn(*args, **kwargs)
        pr.disable()
        s = StringIO.StringIO()
        sortBy = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortBy)
        ps.print_stats()
        print(s.getvalue())
        return retVal


def profileFun(fn):
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retVal = fn(*args, **kwargs)
        pr.disable()
        s = StringIO.StringIO()
        sortBy = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortBy)
        ps.print_stats()
        print(s.getvalue())
        return retVal
    return wrapper

time_log_dict={}


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        time_log_dict[args[0]] = int((te - ts) * 1000)
        # print("{0} takes {1} seconds ".format(args[0],time_log_dict[args[0]]))
        return result
    return timed


# @DecorateProfiler
def MMCsolver(lamda, mu, nservers, mClasses):
    assert sum(lamda/mu) < nservers  # ensure stability

    # initialize \Lamda and \alpha
    lambdaTot = sum(lamda)
    alpha = lamda/lambdaTot

    # create mapping between the combination vectors and matrix columns/rows
    idx_map = dict([(tuple(vect), i)
                    for i,
                    vect in enumerate(generateVectorsFixedSum(mClasses, nservers))])
    # need to use tuple here as 'list' cannot be as a key
    i_map = dict([(idx_map[idx], list(idx)) for idx in idx_map])
    # need to use list here as 'tuple' cannot be modified as will be need further

    # function to get matrix index based on the system state
    def getIndexDict(idx, idx_map):
        try:
            return idx_map[tuple(idx)]
        except KeyError:
            return -1
    # generate matrices A_0 and A_1
    q_max = len(idx_map)
    A0 = np.zeros((q_max, q_max))  # corresponds to terms with i items in queue
    A1 = np.zeros((q_max, q_max))  # corresponds to terms with i+1 items in queue
    for i, idx in i_map.items():
        # diagonal term
        A0[i, i] += 1 + np.sum(idx*mu)/lambdaTot

    # term corresponding to end of service for item j1, start of service for j2
        for j1 in xrange(mClasses):
            for j2 in xrange(mClasses):
                idx[j1] += 1
                idx[j2] -= 1
                i1 = getIndexDict(idx, idx_map)  # convert 'list' back to tuple to use it as a key
                if i1 >= 0:
                    A1[i, i1] += alpha[j2]/lambdaTot*idx[j1]*mu[j1]
                idx[j1] -= 1
                idx[j2] += 1

    # compute matrix Z iteratively
    eps = 0.00000001
    I = np.eye(q_max)  # produces identity matrix
    Z_prev = np.zeros((q_max, q_max))
    delta = 1
    A0_inv = np.linalg.inv(A0)
    while delta > eps:
        Z = np.dot(A0_inv, I + np.dot(A1, np.dot(Z_prev, Z_prev)))  # invA0*(I+A1*Z*Z)
        delta = np.sum(np.abs(Z-Z_prev))
        Z_prev = Z

    # generate Q matrices, it will be stored in a list
    Q = []
    idxMat = []  # matrix with server occupancy for each system state, will be used in computing the system parameters
    Q.insert(0, Z[:])
    idxMat.insert(0, np.array([x for x in i_map.values()]))

    i_map_full = []
    i_map_full.append(i_map)

    # dict([ (tuple(vect), i) for i, vect in enumerate(generateVectorsFixedSum(mClasses, nServers)) ])
    idx_map_nplus = idx_map
    i_map_nplus = i_map  # dict([(idx_map_nplus[idx], list(idx)) for idx in idx_map_nplus ])
    q_max_nplus = len(idx_map_nplus)

    idx_map_n = idx_map_nplus
    i_map_n = i_map_nplus
    q_max_n = q_max_nplus

    A1_n = A1[:]

    for n in range(nservers, 0, -1):
        idx_map_nminus = dict([(tuple(vect), i)
                               for i, vect in enumerate(generateVectorsFixedSum(mClasses, n-1))])
        i_map_nminus = dict([(idx_map_nminus[idx], list(idx)) for idx in idx_map_nminus])
        q_max_nminus = len(idx_map_nminus)

        i_map_full.insert(0, i_map_nminus)

        L_n = np.zeros((q_max_n, q_max_nminus))  # corresponds to terms with i items in queue
        A0_n = np.zeros((q_max_n, q_max_n))  # corresponds to terms with i items in queue
        for i, idx in i_map_n.items():

            # diagonal term
            A0_n[i, i] += 1 + np.sum(idx*mu)/lambdaTot

            # term corresponding to arrival of item item j1
            for j2 in xrange(mClasses):
                idx[j2] -= 1
                i2 = getIndexDict(idx, idx_map_nminus)
                if i2 >= 0:
                    L_n[i, i2] += alpha[j2]
                idx[j2] += 1

        # Q_n = (A_0 - A_1*Q_{n+1})^{-1}*L_n
        Q.insert(0, np.dot(np.linalg.inv(A0_n-np.dot(A1_n, Q[0])), L_n))

        idx_map_nplus = idx_map_n
        i_map_nplus = i_map_n
        q_max_nplus = q_max_n

        idx_map_n = idx_map_nminus
        i_map_n = i_map_nminus
        q_max_n = q_max_nminus
        idxMat.insert(0, np.array([x for x in i_map_n.values()]))

        A1_n = np.zeros((q_max_n, q_max_nplus))  # corresponds to terms with i+1 items in queue
        for i, idx in i_map_n.items():
            # term corresponding to end of service for item j1
            for j1 in xrange(mClasses):
                idx[j1] += 1
                i1 = getIndexDict(idx, idx_map_nplus)
                if i1 >= 0:
                    A1_n[i, i1] += idx[j1]*mu[j1]/lambdaTot
                idx[j1] -= 1

    # compute the P_n for n<k and normalize it such that sum(P_n) = 1
    P = []
    P.append([1.0])

    sm = 1.0
    for n in xrange(nservers):
        P.append(np.dot(Q[n], P[-1]))
        sm += sum(P[-1])

    sm += sum(np.dot(np.linalg.inv(np.eye(len(P[-1])) - Z), np.dot(Z, P[-1])))

    for p in P:
        p[:] /= sm  # normalization

    # compute totals needed for the E[Q_i] - marginal distributions
    inv1minZ = np.linalg.inv(np.eye(len(P[-1])) - Z)
    EQTotal = sum(np.dot(np.dot(np.dot(inv1minZ, inv1minZ), Z), P[-1]))
    EQQmin1Total = 2 * \
        sum(np.dot(np.dot(np.dot(np.dot(np.dot(inv1minZ, inv1minZ), inv1minZ), Z), Z), P[-1]))
    EQ2Total = EQQmin1Total + EQTotal

    # compute 1st and 2nd marginal moments of the numbers in the queue E[Q_i] and E[Q_i^2]
    EQ = alpha*EQTotal
    EQQmin1 = alpha*alpha*EQQmin1Total
    EQ2 = EQQmin1 + EQ

    # compute 1st and 2nd marginal moments of the numbers in the system E[N_i] and E[N_i^2]
    ENTotal = EQTotal + sum(lamda/mu)
    EN = EQ + lamda/mu

    # TODO compute the E[N_i^2]
    ES2 = np.zeros(mClasses)
    for (p, idx) in zip(P[:-1], idxMat[:-1]):
        ES2 += np.dot(p, idx**2)
    ES2 += np.dot(np.dot(inv1minZ, P[-1]), idxMat[-1]**2)

    ESq = alpha*np.dot(np.dot(np.dot(np.dot(inv1minZ, inv1minZ), Z), P[-1]), idxMat[-1])

    EN2 = EQ2 + 2*ESq + ES2

    # compute marginal variances of the numbers in the queue Var[Q_i] and in the system Var[N_i]
    VarQTotal = EQ2Total - EQTotal**2
    VarQ = EQ2 - EQ**2

    VarN = EN2 - EN**2

    # computeMarginalDistributions
    qmax = 1500

    marginalN = np.zeros((mClasses, qmax))

    for m in xrange(mClasses):
        for imap, p in zip(i_map_full[:-1], P[:-1]):
            for i, idx in imap.items():
                marginalN[m, idx[m]] += p[i]

        inv1minAlphaZ = np.linalg.inv(np.eye(len(P[-1])) - (1-alpha[m])*Z)
        frac = np.dot(alpha[m]*Z, inv1minAlphaZ)
        # tmp = np.dot(self.Z, self.P[-1])
        # tmp = np.dot(inv1minAlphaZ, tmp)
        tmp = np.dot(inv1minAlphaZ, P[-1])

        for q in xrange(0, qmax):
            for i, idx in i_map_full[-1].items():
                if idx[m]+q < qmax:
                    marginalN[m, idx[m]+q] += tmp[i]
            tmp = np.dot(frac, tmp)
    return marginalN, EN, VarN


def whittApprox(E1, E2, E3):
    '''
    input: first 3 moments of hyperexpo dist.
    returns: parameters of hyperexpo (p, v1 and v2)
    uses whitt approximation.....
    '''
    x = E1*E3-1.5*E2**2
    # print x
    assert x >= 0.0

    y = E2-2*(E1**2)
    # print y
    assert y >= 0.0

    Ev1 = ((x+1.5*y**2+3*E1**2*y)+math.sqrt((x+1.5*y**2-3*E1**2*y)**2+18*(E1**2)*(y**3)))/(6*E1*y)
    # print Ev1
    assert Ev1 >= 0

    Ev2 = ((x+1.5*y**2+3*E1**2*y)-math.sqrt((x+1.5*y**2-3*E1**2*y)**2+18*(E1**2)*(y**3)))/(6*E1*y)
    assert Ev2 >= 0

    p = (E1-Ev2)/(Ev1-Ev2)
    assert p >= 0

    return 1.0/Ev1, 1.0/Ev2, p


def isServiceRateEqual(mu):
    return len(set(mu)) <= 1


def Approx_MMCsolver(lamda, mu, nServers, mClasses):
    '''
    inputs: lamda->failure rates of SKUs
            mu ->service rates of servers for SKUs
            nServers->number of servers in repairshop
            mClasses->number of SKUs := length of failure rates

    output: Marginal Queue length for each type of SKU
            Expected Queue length  ''  ''   ''   ''
            Variance of Queue length ''  '' ''   ''

    solution: Approximate 3 class system and calls MMCsolver
    '''

    marginalN = []
    EN = []
    VarN = []

    for mCl in range(mClasses):
        # first moment for service time distribution for approximation:
        E_S1 = (np.inner(lamda, 1/mu)-(lamda[mCl]*1/mu[mCl]))/(sum(lamda)-lamda[mCl])  # checked

        # second moment
        E_S2 = 2*(np.inner(lamda, (1/mu)**2) -
                  (lamda[mCl]*(1/mu[mCl])**2))/(sum(lamda)-lamda[mCl])  # checked

        # third moment
        E_S3 = 6*(np.inner(lamda, (1/mu)**3) -
                  (lamda[mCl]*(1/mu[mCl])**3))/(sum(lamda)-lamda[mCl])  # checked

        # calculate inputs for to check neccesity condtion:
        varA = E_S2-E_S1**2
        cA = math.sqrt(varA)/E_S1

        # to check if all of the service rates of approximated service are same
        # if it is true sum of hyperexpo with same parameter is ---> exponential distribution

        mu_copy = []
        mu_copy[:] = mu
        del mu_copy[mCl]

        if isServiceRateEqual(mu_copy) is True:
            # we can assume there is only aggreate remaing streams to one rather than two
            p = 1
            v1 = mu_copy[0]

            lam1 = lamda[mCl]
            # S1=1/mu[mCl]

            lamA1 = p*(sum(lamda)-lamda[mCl])
            # SA1=1/float(v1)

            # if sum(lamda/mu)>nservers:
            #    nservers
            # we have only two streams now so mClasses=2
            marginalLength, ENLength, VarLength = MMCsolver(np.array(
                [lam1, lamA1]), np.array([mu[mCl], v1]), nservers=nServers, mClasses=2)

            marginalN.append(marginalLength[0])
            EN.append(ENLength[0])
            VarN.append(VarLength[0])

        # if (E_S3-(3.0/2.0)*((1+cA**2)**2)*E_S1**3)<0.0:
        # E_S3=(3.0/2.0)*((1+cA**2)**2)*E_S1**3+0.01
        #    print "aaa"
        #    v1, v2, p=whittApprox(E_S1, E_S2, E_S3)

        else:
            # a2 calculation
            a2 = (6*E_S1-(3*E_S2/E_S1))/((6*E_S2**2/4*E_S1)-E_S3)

            # a1 calculation
            a1 = (1/E_S1)+(a2*E_S2/(2*E_S1))

            # v1 calculation
            v1 = (1.0/2.0)*(a1+math.sqrt(a1**2-4*a2))

            # v2 calculation
            v2 = (1.0/2.0)*(a1-math.sqrt(a1**2-4*a2))

            # p calculation
            p = 1-((v2*(E_S1*v1-1))/float((v1-v2)))

            lam1 = lamda[mCl]
            # S1=1/mu[mCl]

            lamA1 = p*(sum(lamda)-lamda[mCl])
            # SA1=1/float(v1)

            lamA2 = (1-p)*(sum(lamda)-lamda[mCl])
            # SA2=1/float(v2)

            # Now we have 3 classes of streams (2 streams for approximation) as usual
            # so mClasses=3

            marginalLength, ENLength, VarLength = MMCsolver(np.array(
                [lam1, lamA1, lamA2]), np.array([mu[mCl], v1, v2]), nservers=nServers, mClasses=3)

            marginalN.append(marginalLength[0])
            EN.append(ENLength[0])
            VarN.append(VarLength[0])

    return marginalN, EN, VarN


def Approx_MMCsolver2(lamda, mu, nservers, mClasses):
    '''
    inputs: lamda->failure rates of SKUs
            mu ->service rates of servers for SKUs
            nservers->number of servers in repairshop
            mClasses->number of SKUs := length of failure rates

    output: Marginal Queue length for each type of SKU
            Expected Queue length  ''  ''   ''   ''
            Variance of Queue length ''  '' ''   ''

    solution: Approximate 3 class system and calls MMCsolver
    '''

    # print nservers
    marginalN = []
    EN = []
    VarN = []

    for mCl in range(mClasses):
        # first moment for service time distribution for approximation:
        E_S1 = (np.inner(lamda, 1/mu)-(lamda[mCl]*1/mu[mCl]))/(sum(lamda)-lamda[mCl])  # checked
        # print E_S1
        # second moment
        E_S2 = 2*(np.inner(lamda, (1/mu)**2) -
                  (lamda[mCl]*(1/mu[mCl])**2))/(sum(lamda)-lamda[mCl])  # checked

        # third moment
        E_S3 = 6*(np.inner(lamda, (1/mu)**3) -
                  (lamda[mCl]*(1/mu[mCl])**3))/(sum(lamda)-lamda[mCl])  # checked

        # calculate inputs for to check neccesity condtion:
        varA = E_S2-E_S1**2
        cA = math.sqrt(varA)/E_S1

        assert (E_S3-(3.0/2.0)*((1+cA**2)**2)*E_S1**3) > 0

        # to check if all of the service rates of approximated service are same
        # if it is true sum of hyperexpo with same parameter is ---> exponential distribution

        mu_copy = []
        mu_copy[:] = mu
        del mu_copy[mCl]

        if isServiceRateEqual(mu_copy) is True:
            # we can assume there is only aggreate remaing streams to one rather than two
            p = 1
            v1 = mu_copy[0]

            lam1 = lamda[mCl]
            # S1=1/mu[mCl]

            lamA1 = p*(sum(lamda)-lamda[mCl])
            # SA1=1/float(v1)

            # sum(lamda/mu)<nservers

            if sum(np.array([lam1, lamA1])/np.array([mu[mCl], v1])) > nservers:
                # print "hasan"
                nservers = int(sum(np.array([lam1, lamA1])/np.array([mu[mCl], v1])))+1

            # we have only two streams now so mClasses=2
            marginalLength, ENLength, VarLength = MMCsolver(np.array(
                [lam1, lamA1]), np.array([mu[mCl], v1]), nservers, mClasses=2)

            marginalN.append(marginalLength[0])
            EN.append(ENLength[0])
            VarN.append(VarLength[0])
            # print "aaaa"

        # if (E_S3-(3.0/2.0)*((1+cA**2)**2)*E_S1**3)<0.0:
        # E_S3=(3.0/2.0)*((1+cA**2)**2)*E_S1**3+0.01
        #    print "aaa"
        #    v1, v2, p=whittApprox(E_S1, E_S2, E_S3)

        else:

            v1, v2, p = whittApprox(E_S1, E_S2, E_S3)
            # print v1
            # print v2

            lam1 = lamda[mCl]
            # S1=1/mu[mCl]

            lamA1 = p*(sum(lamda)-lamda[mCl])
            # SA1=1/float(v1)

            lamA2 = (1-p)*(sum(lamda)-lamda[mCl])
            # SA2=1/float(v2)

            if sum(np.array([lam1, lamA1, lamA2])/np.array([mu[mCl], v1, v2])) >= nservers:
                # print "turan"
                nservers = int(sum(np.array([lam1, lamA1, lamA2])/np.array([mu[mCl], v1, v2])))+1
            # Now we have 3 classes of streams (2 streams for approximation) as usual
            # so mClasses=3

            marginalLength, ENLength, VarLength = MMCsolver(np.array(
                [lam1, lamA1, lamA2]), np.array([mu[mCl], v1, v2]), nservers, mClasses=3)

            marginalN.append(marginalLength[0])
            EN.append(ENLength[0])
            VarN.append(VarLength[0])

    return marginalN, EN, VarN, nservers


# code for optimization inventories after given queue length distribution
def OptimizeStockLevelsAndCosts(holdingCosts, penalty, marginalDistribution):

    if not isinstance(holdingCosts, np.ndarray):
        holdingCosts = np.array(holdingCosts)

    if len(marginalDistribution.shape) == 1:
        marginalDistribution = marginalDistribution.reshape(1, len(marginalDistribution))

    nSKUs = len(holdingCosts)
    maxQueue = marginalDistribution.shape[1]
    n_array = np.array(range(maxQueue))
    S = np.zeros(nSKUs, dtype=int)
    PBO = np.sum(marginalDistribution[:, 1:], axis=1)
    EBO = np.sum(marginalDistribution*np.array(range(marginalDistribution.shape[1])), axis=1)

    hb_ratio = holdingCosts/penalty
    for sk in xrange(nSKUs):
        while S[sk] < maxQueue and np.sum(marginalDistribution[sk, S[sk]+1:]) > hb_ratio[sk]:
            S[sk] += 1
            # -= marginalDistribution[sk, S[sk]]
            PBO[sk] = np.sum(marginalDistribution[sk, S[sk]+1:])
            EBO[sk] = np.sum(marginalDistribution[sk, S[sk]:]*n_array[:-S[sk]])  # -= PBO[sk]

    totalCost = np.sum(S*holdingCosts) + np.sum(penalty*EBO)
    hCost = np.sum(S*holdingCosts)
    pCost = np.sum(penalty*EBO)
    # print ((EBO < 0).sum() == EBO.size).astype(np.int)
    # if pCost<0.0:
    #    print EBO
    # print  ((EBO < 0).sum() == EBO.size).astype(np.int)
    # print all(i >= 0.0 for i in marginalDistribution)

    return totalCost, hCost, pCost, S, EBO


def individual2cluster(individual):
    '''
    -input: list of integers representing assingment of SKUs to clusters
    -output: list of list representing clusters and assinged SKUs in each cluster
    '''
    return [[i + 1 for i, j in enumerate(individual) if j == x] for x in set(individual)]


def evalOneMax(FailureRates, ServiceRates, holding_costs, penalty_cost, skillCost, machineCost, individual):
    '''
    input: -Individual representing clustering scheme
           -Failure rates and corresponding service rates of each SKU
           -Related cost terms holding costs for SKUs(array), backorder, skill and server (per server and per skill)
           -MMCsolver and Approx_MMCsolver functions--> to find Queue length dist. of failed SKUs
                                                    --> number of SKUS >=4 use approximation

           -OptimizeStockLevels calculates EBO and S for giving clustering (Queue length dist.)

     output: Returns best total cost and other cost terms, Expected backorder (EBO) and stocks (S) for each SKU, # of
             servers at each cluster

     evalOneMax function evaluates the fitness of individual chromosome by:
           (1) chromosome converted a clustering scheme
           (2) for each SKU in each cluster at the clustering scheme queue length dist. evaluated by calling MMC solver
           (3) OptimzeStockLevels function is called by given queue length dist. and initial costs are calculated
           (4) Local search is performed by increasing server numbers in each cluster by one and step (2) and (3) repetead
           (5) Step (4) is repated if there is a decrease in total cost


    Warning !! type matching array vs list might be problem (be careful about type matching)

    '''

    # from individual to cluster
    cluster_GA = individual2cluster(individual)
    # bestCost=float('inf')
    # bestCluster=[]
    # print "\n"
    # print individual
    # print cluster_GA

    bestS = []
    bestEBO = []
    EBO_cluster = []
    S_cluster = []
    bestserverAssignment = []
    serverAssignment = []
    sliceIndex2 = []
    TotalCost = 0.0
    TotalHolding, TotalPenalty, TotalSkillCost, TotalMachineCost = 0.0, 0.0, 0.0, 0.0
    # LogFileList=[]
    # logFile={}
    # iterationNum=0
    for cluster in cluster_GA:
        sliceIndex2[:] = cluster
        sliceIndex2[:] = [x - 1 for x in sliceIndex2]

        sRate = np.array(ServiceRates[sliceIndex2])
        fRate = np.array(FailureRates[sliceIndex2])
        hcost = np.array(holding_costs[sliceIndex2])

        min_nserver = int(sum(fRate/sRate))+1
        # print sliceIndex2
        # print "RUn FINISHED \n"
        # sys.exit(0)
        # costTemp=0
        # while costTemp<=machineCost:
        if len(sRate) <= 3:
            marginalDist, EN, VarN = MMCsolver(fRate, sRate, min_nserver, len(fRate))
        else:
            marginalDist, EN, VarN, min_nserverUpdate = Approx_MMCsolver2(
                fRate, sRate, min_nserver, len(fRate))
            min_nserver = min_nserverUpdate

        totalCostClust, hCost, pCost, S, EBO = OptimizeStockLevelsAndCosts(
            hcost, penalty_cost, np.array(marginalDist))

        # increasing number of servers and checking if total cost decreases

        TotalMachine_Cost = min_nserver*machineCost
        TotalSkill_Cost = min_nserver*len(fRate)*skillCost

        totalCostClust = totalCostClust+TotalMachine_Cost+TotalSkill_Cost

        while True:
            min_nserver += 1
            if len(sRate) <= 3:
                marginalDist, EN, VarN = MMCsolver(fRate, sRate, min_nserver, len(fRate))
            else:
                marginalDist, EN, VarN, min_nserverUpdate = Approx_MMCsolver2(
                    fRate, sRate, min_nserver, len(fRate))
                min_nserver = min_nserverUpdate

            temp_totalCostClust, temp_hCost, temp_pCost, temp_S, temp_EBO = OptimizeStockLevelsAndCosts(
                hcost, penalty_cost, np.array(marginalDist))
            temp_TotalMachine_Cost = min_nserver*machineCost
            temp_TotalSkill_Cost = min_nserver*len(fRate)*skillCost

            temp_totalCostClust = temp_totalCostClust+temp_TotalMachine_Cost+temp_TotalSkill_Cost

            if temp_totalCostClust > totalCostClust:
                min_nserver -= 1
                break
            else:
                totalCostClust = temp_totalCostClust

                TotalMachine_Cost = temp_TotalMachine_Cost
                TotalSkill_Cost = temp_TotalSkill_Cost
                hCost = temp_hCost
                pCost = temp_pCost

        TotalHolding += hCost
        TotalPenalty += pCost

        TotalSkillCost += TotalSkill_Cost
        TotalMachineCost += TotalMachine_Cost

        TotalCost = TotalCost+totalCostClust

        EBO_cluster.append(EBO.tolist())
        S_cluster.append(S.tolist())
        serverAssignment.append(min_nserver)

    return TotalCost,

# bestHolding, bestPenalty, bestMachineCost, bestSkillCost, bestCluster, bestS, bestEBO, \
#            bestserverAssignment, LogFileList
# DONT FORGET COME AT THE END!!!


def Final_evalOneMax(FailureRates, ServiceRates, holding_costs, penalty_cost, skillCost, machineCost, individual):
    '''
    input: -Individual representing clustering scheme
           -Failure rates and corresponding service rates of each SKU
           -Related cost terms holding costs for SKUs(array), backorder, skill and server (per server and per skill)
           -MMCsolver and Approx_MMCsolver functions--> to find Queue length dist. of failed SKUs
                                                    --> number of SKUS >=4 use approximation

           -OptimizeStockLevels calculates EBO and S for giving clustering (Queue length dist.)

     output: Returns best total cost and other cost terms, Expected backorder (EBO) and stocks (S) for each SKU, # of
             servers at each cluster

     evalOneMax function evaluates the fitness of individual chromosome by:
           (1) chromosome converted a clustering scheme
           (2) for each SKU in each cluster at the clustering scheme queue length dist. evaluated by calling MMC solver
           (3) OptimzeStockLevels function is called by given queue length dist. and initial costs are calculated
           (4) Local search is performed by increasing server numbers in each cluster by one and step (2) and (3) repeted
           (5) Step (4) is repated if there is a decrease in total cost


    Warning !! type matching array vs list might be problem (be careful about type matching)

    '''

    # from individual to cluster
    cluster_GA = individual2cluster(individual)
    # bestCost=float('inf')
    # bestCluster=[]
    bestS = []
    bestEBO = []
    EBO_cluster = []
    S_cluster = []
    bestserverAssignment = []
    serverAssignment = []
    sliceIndex2 = []
    TotalCost = 0.0
    TotalHolding, TotalPenalty, TotalSkillCost, TotalMachineCost = 0.0, 0.0, 0.0, 0.0
    # LogFileList=[]
    # logFile={}
    # iterationNum=0
    for cluster in cluster_GA:
        sliceIndex2[:] = cluster
        sliceIndex2[:] = [x - 1 for x in sliceIndex2]

        sRate = np.array(ServiceRates[sliceIndex2])
        fRate = np.array(FailureRates[sliceIndex2])
        hcost = np.array(holding_costs[sliceIndex2])

        min_nserver = int(sum(fRate/sRate))+1

        # costTemp=0
        # while costTemp<=machineCost:
        if len(sRate) <= 3:
            marginalDist, EN, VarN = MMCsolver(fRate, sRate, min_nserver, len(fRate))
        else:
            marginalDist, EN, VarN, min_nserverUpdate = Approx_MMCsolver2(
                fRate, sRate, min_nserver, len(fRate))
            min_nserver = min_nserverUpdate

        totalCostClust, hCost, pCost, S, EBO = OptimizeStockLevelsAndCosts(
            hcost, penalty_cost, np.array(marginalDist))

        # increasing number of servers and checking if total cost decreases

        TotalMachine_Cost = min_nserver*machineCost
        TotalSkill_Cost = min_nserver*len(fRate)*skillCost

        totalCostClust = totalCostClust+TotalMachine_Cost+TotalSkill_Cost

        while True:
            min_nserver += 1
            if len(sRate) <= 3:
                marginalDist, EN, VarN = MMCsolver(fRate, sRate, min_nserver, len(fRate))
            else:
                marginalDist, EN, VarN, min_nserverUpdate = Approx_MMCsolver2(
                    fRate, sRate, min_nserver, len(fRate))
                min_nserver = min_nserverUpdate

            temp_totalCostClust, temp_hCost, temp_pCost, temp_S, temp_EBO = OptimizeStockLevelsAndCosts(
                hcost, penalty_cost, np.array(marginalDist))
            temp_TotalMachine_Cost = min_nserver*machineCost
            temp_TotalSkill_Cost = min_nserver*len(fRate)*skillCost

            temp_totalCostClust = temp_totalCostClust+temp_TotalMachine_Cost+temp_TotalSkill_Cost

            if temp_totalCostClust > totalCostClust:
                min_nserver -= 1
                break
            else:
                totalCostClust = temp_totalCostClust

                TotalMachine_Cost = temp_TotalMachine_Cost
                TotalSkill_Cost = temp_TotalSkill_Cost
                hCost = temp_hCost
                pCost = temp_pCost

        TotalHolding += hCost
        TotalPenalty += pCost

        TotalSkillCost += TotalSkill_Cost
        TotalMachineCost += TotalMachine_Cost

        TotalCost = TotalCost+totalCostClust

        EBO_cluster.append(EBO.tolist())
        S_cluster.append(S.tolist())
        serverAssignment.append(min_nserver)

    return TotalCost, TotalHolding, TotalPenalty, TotalMachineCost, TotalSkillCost, cluster_GA, S_cluster, EBO_cluster, serverAssignment
# DONT FORGET COME AT THE END!!!


def swicthtoOtherMutation(individual, indpb):
    '''
    input- individual chromosome
    output- some genes changed to other genes in chromosome (changing clusters)
    There might be other ways of mutation - swaping clusters of two SKUs (crossover does that) two way swap
                                          - opening a new cluster
                                          - closing a cluster and allocated SKUs in that cluster to another cluster
                                          -(local or tabu search idea!!)
    '''
    # to keep orginal probabilty of switching to other cluster during iteration
    individual_copy = individual[:]
    for i in range(len(individual)):
        if random.random() <= indpb:
            if random.random() <= 1.5:  # switch only version _v4a
                # set is used to give equal probability to assign any other cluster
                # without set there is a higher probablity to assigning to a cluster that inclludes more SKUs
                if len(list(set(individual_copy).difference(set([individual_copy[i]])))) >= 1:
                    individual[i] = random.choice(
                        list(set(individual_copy).difference(set([individual_copy[i]]))))

            else:
                # This mutation type aimed for generating new cluster and going beyond the allowed maximum num cluster
                if len(list(set(range(1, len(individual_copy)+1)).difference(set(individual_copy)))) >= 1:
                    individual[i] = random.choice(
                        list(set(range(1, len(individual_copy)+1)).difference(set(individual_copy))))

    return individual

# there is infeasibility
# a=['a',  'a']
# random.choice(list(set(a).difference(a[0])))
# list(set(a))
# random.choice(list(set(a).difference(a[0])))

def ns_two_way_swap(S):
    S_new = S[:]
    numSKUs = len(S_new)
    idx1, idx2 = random.sample(range(0, numSKUs), 2)
    S_new[idx1] = S[idx2]
    S_new[idx2] = S[idx1]
    return S_new

def ns_shuffle(S):
    S_new = S[:]
    random.shuffle(S_new)
    return S_new

def ns_mutate_random(minCluster, maxCluster, S, n=1):
    S_new = S[:]
    numSKUs = len(S_new)
    idx1, idx2 = random.sample(range(0, numSKUs), 2)
    ex_cluster_number = S[idx1]
    numbers = range(minCluster, ex_cluster_number) + range(ex_cluster_number + 1, maxCluster)
    S_new[idx1] = random.choice(numbers)
    if n == 2:
        ex_cluster_number = S[idx2]
        numbers = range(minCluster, ex_cluster_number) + range(ex_cluster_number + 1, maxCluster)
        S_new[idx2] = random.choice(numbers)
    return S_new

def mixed(minCluster, maxCluster, S):
    r = random.uniform(0, 1)
    if r <= 0.25:
        return ns_two_way_swap(S)
    if r <= 0.5:
        return ns_shuffle(S)
    if r <= 0.75:
        return ns_mutate_random(minCluster, maxCluster, S)
    return ns_mutate_random(minCluster, maxCluster, S, 2)

def neighborhood_solution(S, minCluster, maxCluster, method=5):
    if method ==5:
        return mixed(minCluster, maxCluster, S)
    if method ==1:
        return ns_two_way_swap(S)
    if method == 2:
        return ns_shuffle(S)
    if method ==   3:
        return ns_mutate_random(minCluster, maxCluster, S)
    return ns_mutate_random(minCluster, maxCluster, S, 2)


def getHigherEnergyValuedNeighbors(pop0, obj_func, i, minCluster, maxCluster):
    # create neighbor individuals
        ind = pop0[i]
        # find positive neighbors (with higher energy values)
        while True:
            neighbor_i = neighborhood_solution(ind, minCluster, maxCluster)
            fitnesVal_ind = obj_func(ind)
            fitnesVal_neig = obj_func(neighbor_i)
            if fitnesVal_neig > fitnesVal_ind:
                return i, neighbor_i

def check_neighbor_solutions(S, T, TC_S_best, I_sa_inner, obj_func, process_id, minCluster, maxCluster):
    # print ("ProcessId:{0} Individual{1} TC_bset:{2}".format(process_id, S, TC_S_best))
    tc_s_best_ = TC_S_best
    s_best_ = S
    Nt_inner=0
    while Nt_inner <= I_sa_inner:
        s_prime = neighborhood_solution(S, minCluster, maxCluster)
        # evaluate the costs
        tc_s = obj_func(S)[0]
        tc_sprime = obj_func(s_prime)[0]
        delta_e = tc_sprime - tc_s
        if delta_e <= 0:
            S = s_prime
            tc_s = tc_sprime
            if tc_s < tc_s_best_:
                print("Successfull Process {0} TC:{1} InnerIter:{2}".format(process_id, tc_s, Nt_inner))
                tc_s_best_ = tc_s
                s_best_ = S
                Nt_inner = 0
        else:
            r = random.uniform(0, 1)
            if r < math.exp(-delta_e / T):
                # allow  bad solution
                S = s_prime
                tc_s = tc_sprime
        Nt_inner += 1

    # print("Nonsuccessfull Thread")
    if TC_S_best > tc_s_best_:
        return s_best_, tc_s_best_
    return S, tc_s

#@profileFun
def GAPoolingHeuristic(case_id, failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost, numSKUs, minCluster, maxCluster):


    # 1 is for maximization -1 for minimization
    # Minimize total cost
    creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    def generateIndividual(numSKUs, minCluster, maxCluster):

        # Generating initial indvidual that are in the range of given max-min cluster numbers

        individual = [0]*numSKUs

        randomSKUsindex = np.random.choice(range(numSKUs), minCluster, replace=False)
        cluster_randomSKUs = np.random.choice(range(1, maxCluster+1), minCluster, replace=False)

        for i in range(minCluster):
            individual[randomSKUsindex[i]] = cluster_randomSKUs[i]

        for i in range(numSKUs):
            if individual[i] == 0:
                individual[i] = random.randint(1, maxCluster)

    # print type (creator.Individual(individual))
        return creator.Individual(individual)

    toolbox = base.Toolbox()

    # Attribute generator
    #                      define 'attr_bool' to be an attribute ('gene')
    #                      which corresponds to integers sampled uniformly
    #                      from the range [1,number of SKUs] (i.e. 0 or 1 with equal
    #                      probability)

    # Structure initializers
    #                         define 'individual' to be an individual
    #                         consisting of #number of maximum cluster =#of SKUs 'attr_bool' elements ('genes')
    toolbox.register("individual", generateIndividual, numSKUs, minCluster, maxCluster)

    # define the population to be a list of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # the goal ('fitness') function to be maximized
    # for objective function call pooling optimizer !!!
    # what values need for optimizer !!!

    # def evalOneMax(individual):
    #    return sum(individual),

    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    toolbox.register("evaluate", evalOneMax, failure_rates, service_rates,
                     holding_costs, penalty_cost, skill_cost, machine_cost)

    # register the crossover operator
    toolbox.register("mate", tools.cxTwoPoint)

    # register a mutation operator with a probability to
    # flip each attribute/gene of 0.05
    #
    toolbox.register("mutate", swicthtoOtherMutation, indpb=0.4)
    # toolbox.register("mutate", swicthtoOtherMutation)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.

    toolbox.register("select", tools.selTournament, tournsize=10)

    random.seed(64)

    # create an initial population of 100 individuals (where
    # each individual is a list of integers)

    # print("Start of evolution")
    pop = toolbox.population(n=1)

    # helper for findT0
    def x_bar(T, tcs):
        tot_next = 0.0
        tot_prev = 0.0
        nsamples = len(tcs) / 2
        for i in range(nsamples):
            tot_next = tot_next + math.exp(-tcs[i + nsamples] / T)
            tot_prev = tot_prev + math.exp(-tcs[i] / T)
        return tot_next / tot_prev

    def findT0(Tn, tc, X0, eps, p):
        x_bar_val = x_bar(Tn, tc)
        while math.fabs(x_bar_val - X0) > eps:
            Tn = Tn * ((math.log(x_bar_val) / math.log(X0)) ** (1.0 / p))
            x_bar_val = x_bar(Tn, tc)
            # print("diff:{0}, eps:{1}".format(math.fabs(x_bar_val - X0), eps))
        return Tn

    def findT1(tc, nsamples, X0):
        tot = 0.0
        for i in range(nsamples):
            tot = tot + tc[i + nsamples] - tc[i]
        return -tot / (nsamples * math.log(X0))


    """
    Finds best candidates.

    :param T0: Initial Tempreture
    :param Tf: Final Temprature.
    :param I_sa_inner: Max num of non-improved solution in inside loop.
    :param I_sa_main: Max num of non-improved solution in main loop.

    :returns: A list of selected individuals.
    """
    _ , case_no = case_id.split(':')
    case_no = int(case_no)

    def MSSA(T0, Tf=10, I_sa_inner = 12, I_sa_main = 12.0, Psize=5):

        S_list = n_best_individuals[case_no-1]
        print ("Best {0} individuals of KMedian are: {1}".format(Psize, S_list))
        # get their fitness(obj) values
        fitness_vals = list(map(toolbox.evaluate, S_list))
        # find best total cost
        TC_S_best = min(fitness_vals)
        # find best individual index
        best_idx = fitness_vals.index(TC_S_best)
        T = T0
        S = S_list[best_idx]
        S_best = S
        NT_main = 1   # corresponds to N in paper
        iter_idx = 0  # logging purposes
        # TRICK
        # set all other solutions to the best solution
        S_list = [S_best] * Psize

        # print("Initial Best Cost:{0} Best Individual{1}".format(TC_S_best,S_best))
        (a, f) = (3, 0.25) #nonlinear cooling parameters
        # start_time = time.time()  # remember when we started
        while NT_main <= I_sa_main and T >= Tf:
            # traverse each individual
            # traverse each individual
            # start the search operations and mark each future with individuals array index
            with cf.ProcessPoolExecutor(max_workers=5) as executor:
                future_to_k = {executor.submit(check_neighbor_solutions,
                                               S_list[k],
                                               T,
                                               TC_S_best,
                                               I_sa_inner,
                                               toolbox.evaluate,
                                               k,
                                               minCluster,
                                               maxCluster) : k for k in range(Psize)}
                # executor.shutdown()
                # future result is (S,TC_Cost) pair
                results = []
                for future in cf.as_completed(future_to_k):
                    results.append(future.result())

                # print("Case:{0} Results returned:{1}".format(case_no, len(results)))
                # sort the results by Total cost
                sorted_results = sorted(results, key=lambda x: x[1])
                best_result = sorted_results[0]
                if best_result[1] < TC_S_best:
                    S_best = best_result[0]
                    TC_S_best = best_result[1]
                    print("New Best! T:{0} TC_Best:{1} Ind:{2}".format(T, TC_S_best, S_best))
                    NT_main = 1
                    # set all other solutions to the best solution
                    S_list = [S_best] * Psize
                else:
                    NT_main += 1

            # use a nonlinear cooling method
            P = math.log(math.log(T0 / Tf) / math.log(a))
            Q = math.log(1.0 / f)
            b = P / Q
            T = T0 * pow(a, -pow(NT_main/(f*I_sa_main), b))
            iter_idx += 1
            print("Iter:{0} Cooling... T={1} TC_Best:{2}".format(iter_idx, T, TC_S_best))
        return (S_best, TC_S_best)

    # num_samples_arr = [20, 100, 500]
    num_samples_arr = [40]
    # X0_arr = [0.9, 0.8, 0.5]
    X0_arr = [0.9]
    eps = 0.01
    p = 1.0

    for num_samples in num_samples_arr:
        pop0 = toolbox.population(n=num_samples * 2)
        print("Finding Neighbors to find T0 for Case:{0}".format(case_no))
        with cf.ProcessPoolExecutor(max_workers=5) as exctr:
            future_to_i = {exctr.submit(getHigherEnergyValuedNeighbors, pop0,
                                        toolbox.evaluate, i, minCluster, maxCluster): i for i in range(num_samples)}
            # future result is (i,neighbori) pair
            for ftr in cf.as_completed(future_to_i):
                i, negh_i = ftr.result()
                pop0[i+num_samples] = creator.Individual(negh_i)

        # Evaluate the entire population
        TCs = list(map(toolbox.evaluate, pop0))
        for ind, tc in zip(pop0, TCs):
            ind.fitness.values = tc
        # Extracting all the fitnesses of
        tc = [ind.fitness.values[0] for ind in pop0]
        for x in range(num_samples):
            if(tc[x] > tc[x+num_samples]):
                print ("Neighbor Solution Fails for T0")
        print ("Finding T0")
        for X0 in X0_arr:
            T1 = findT1(tc, num_samples, X0)
            T0 = findT0(T1, tc, X0, eps, p)
            print("Running MSSA")
            S_best, TC_S_best = MSSA(T0=T0, Psize=5)
            print("Best Solution:{0} Best Total Cost:{1}".format(S_best, TC_S_best))


    pop[0] = creator.Individual(S_best)
    TCs = list(map(toolbox.evaluate, pop))
    for ind, tc in zip(pop, TCs):
        ind.fitness.values = tc
    best_ind = tools.selBest(pop, 1)[0]
    # print("Best individual is %s, %s" % (individual2cluster(best_ind), best_ind.fitness.values))
    return best_ind.fitness.values, best_ind


# file = "C:/Users/Fuat/Dropbox/Pooling_GA/fullRangeResultsFullFlexNew.json"
file = "/home/atmis/Projects/Python/kmedian_final_version/fullRangeResultsFullFlexNew.json"
json_case = [json.loads(line) for line in open(file, "r")]
sorted_assignments = sorted(json_case, key=operator.itemgetter('caseID'))
json_cases = sorted_assignments[:]
for case in json_cases:
    print (case["caseID"])

# RUN of ALgorithm STARTS HERE ##
# json_case
results = []
GAPoolingResult = {}
case_idx = 0

# get best n individuals found by kmedian algorithm
num_cases = len(json_case)-1
n_best_individuals = get_n_best_individuals(num_cases+1, n=5)


for case in json_cases:
    if case["caseID"] not in ["Case: 0128"]:
         continue
    if case["caseID"] != "Case: 000x":
        failure_rates = case["failure_rates"]
        service_rates = case["service_rates"]
        holding_costs = case["holding_costs"]
        skill_cost = case["skill_cost"]
        penalty_cost = case["penalty_cost"]
        machine_cost = case["machine_cost"]
    # print (case["caseID"], " is runnig")
    start_time = time.time()

    # print len(failure_rates), failure_rates
    # unrestricted initial population _v4a
    numSKUs, minCluster, maxCluster = len(failure_rates), 1, len(failure_rates)

    _, best_ind = GAPoolingHeuristic(case["caseID"], np.array(failure_rates), np.array(service_rates),
                                     np.array(holding_costs), penalty_cost, skill_cost, machine_cost, numSKUs, minCluster, maxCluster)
    stop_time = time.time() - start_time
    # best individual is ran one more the for statistical data collection and recording
    # Using Final_evalOneMax
    bestCost, bestHolding, bestPenalty, bestMachineCost, bestSkillCost, bestCluster, bestS, bestEBO, bestserverAssignment = Final_evalOneMax(
        np.array(failure_rates), np.array(service_rates), np.array(holding_costs), penalty_cost, skill_cost, machine_cost, best_ind)

    GAPoolingResult["caseID"] = case["caseID"]

    GAPoolingResult["GAPoolingruntime"] = stop_time
    GAPoolingResult["GAPoolingTotalCost"] = bestCost
    GAPoolingResult["GAPoolingHoldingCost"] = bestHolding
    GAPoolingResult["GAPoolingPenaltyCost"] = bestPenalty
    GAPoolingResult["GAPoolingMachineCost"] = bestMachineCost
    GAPoolingResult["GAPoolingSkillCost"] = bestSkillCost

    GAPoolingResult["GAPoolingCluster"] = bestCluster
    GAPoolingResult["GAPoolingS"] = bestS
    GAPoolingResult["GAPoolingEBO"] = bestEBO
    GAPoolingResult["GAPoolingServerAssignment"] = bestserverAssignment
    # KmedianResult["KmedianLogFile"]=LogFileList

    GAPoolingResult["GAP"] = bestCost-case["total_cost"]
    GAPoolingResult["GAPoolingPercentGAP"] = 100 * \
        (bestCost-case["total_cost"])/case["total_cost"]

    GAPoolingResult["simulationGAresults"] = case
    results.append(GAPoolingResult)
    case_idx += 1
    with open('results_p4.csv', 'a') as csvfile:
        fieldnames = ['case_id', 'running_time', 'total_cost', 'holding_cost',
                      'penalty_cost', 'machine_cost', 'skill_cost', 'best_cluster',
                      'bestS', 'bestEBO', 'bestServerAssignment', 'GAP', 'GAPoolingPercentGAP']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if case_idx == 1:
            writer.writeheader()
        writer.writerow({'case_id': case["caseID"],
                         'running_time': stop_time, 'total_cost': bestCost,
                         'holding_cost': bestHolding, 'penalty_cost': bestPenalty,
                         'machine_cost': bestMachineCost, 'skill_cost': bestSkillCost,
                         'best_cluster': bestCluster, 'bestS': bestS, 'bestEBO': bestEBO,
                         'bestServerAssignment': bestserverAssignment, 'GAP': GAPoolingResult["GAP"],
                         'GAPoolingPercentGAP': GAPoolingResult["GAPoolingPercentGAP"]})

    GAPoolingResult = {}

# Results are recorder as json file
with open('GAPoolingAll_v4a_p4.json', 'w') as outfile:
    json.dump(results, outfile)
