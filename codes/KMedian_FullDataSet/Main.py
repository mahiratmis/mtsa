import random
import numpy as np
import math
import sys
import itertools
import json
import sframe as sf
from bokeh.charts import BoxPlot, output_notebook, show
#import seaborn as sns
#import pulp
from pulp import *
import time
from gurobipy import *

# Define the generator that will generate all possible assignments of classes to servers, without permutations
def generateVectorsFixedSum(m,n):
    # generator for all combinations of $w$ for given number of servers and classes
    if m==1:
        yield [n]
    else:
        for i in range(n+1):
            for vect in generateVectorsFixedSum(m-1,n-i):
                yield [i]+vect


def MMCsolver(lamda, mu, nservers, mClasses):
    assert sum(lamda / mu) < nservers  # ensure stability

    # initialize \Lamda and \alpha
    lambdaTot = sum(lamda)
    alpha = lamda / lambdaTot

    # create mapping between the combination vectors and matrix columns/rows
    idx_map = dict([(tuple(vect), i) for i, vect in enumerate(generateVectorsFixedSum(mClasses, nservers))])
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
        A0[i, i] += 1 + np.sum(idx * mu) / lambdaTot

        # term corresponding to end of service for item j1, start of service for j2
        for j1 in xrange(mClasses):
            for j2 in xrange(mClasses):
                idx[j1] += 1;
                idx[j2] -= 1
                i1 = getIndexDict(idx, idx_map)  # convert 'list' back to tuple to use it as a key
                if i1 >= 0: A1[i, i1] += alpha[j2] / lambdaTot * idx[j1] * mu[j1]
                idx[j1] -= 1;
                idx[j2] += 1

    # compute matrix Z iteratively
    eps = 0.00000001
    I = np.eye(q_max)  # produces identity matrix
    Z_prev = np.zeros((q_max, q_max))
    delta = 1
    A0_inv = np.linalg.inv(A0)
    while delta > eps:
        Z = np.dot(A0_inv, I + np.dot(A1, np.dot(Z_prev, Z_prev)))  # invA0*(I+A1*Z*Z)
        delta = np.sum(np.abs(Z - Z_prev))
        Z_prev = Z

    # generate Q matrices, it will be stored in a list
    Q = []
    idxMat = []  # matrix with server occupancy for each system state, will be used in computing the system parameters
    Q.insert(0, Z[:])
    idxMat.insert(0, np.array([x for x in i_map.values()]))

    i_map_full = []
    i_map_full.append(i_map)

    idx_map_nplus = idx_map  # dict([ (tuple(vect), i) for i, vect in enumerate(generateVectorsFixedSum(mClasses, nServers)) ])
    i_map_nplus = i_map  # dict([(idx_map_nplus[idx], list(idx)) for idx in idx_map_nplus ])
    q_max_nplus = len(idx_map_nplus)

    idx_map_n = idx_map_nplus
    i_map_n = i_map_nplus
    q_max_n = q_max_nplus

    A1_n = A1[:]

    for n in range(nservers, 0, -1):
        idx_map_nminus = dict([(tuple(vect), i) for i, vect in enumerate(generateVectorsFixedSum(mClasses, n - 1))])
        i_map_nminus = dict([(idx_map_nminus[idx], list(idx)) for idx in idx_map_nminus])
        q_max_nminus = len(idx_map_nminus)

        i_map_full.insert(0, i_map_nminus)

        L_n = np.zeros((q_max_n, q_max_nminus))  # corresponds to terms with i items in queue
        A0_n = np.zeros((q_max_n, q_max_n))  # corresponds to terms with i items in queue
        for i, idx in i_map_n.items():

            # diagonal term
            A0_n[i, i] += 1 + np.sum(idx * mu) / lambdaTot

            # term corresponding to arrival of item item j1
            for j2 in xrange(mClasses):
                idx[j2] -= 1
                i2 = getIndexDict(idx, idx_map_nminus)
                if i2 >= 0: L_n[i, i2] += alpha[j2]
                idx[j2] += 1

        # Q_n = (A_0 - A_1*Q_{n+1})^{-1}*L_n
        Q.insert(0, np.dot(np.linalg.inv(A0_n - np.dot(A1_n, Q[0])), L_n))

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
                if i1 >= 0: A1_n[i, i1] += idx[j1] * mu[j1] / lambdaTot
                idx[j1] -= 1

    # compute the P_n for n<k and normalize it such that sum(P_n) = 1
    P = []
    P.append([1.0])

    sm = 1.0
    for n in xrange(nservers):
        P.append(np.dot(Q[n], P[-1]))
        sm += sum(P[-1])

    sm += sum(np.dot(np.linalg.inv(np.eye(len(P[-1])) - Z), np.dot(Z, P[-1])))

    for p in P: p[:] /= sm  # normalization

    # compute totals needed for the E[Q_i] - marginal distributions
    inv1minZ = np.linalg.inv(np.eye(len(P[-1])) - Z)
    EQTotal = sum(np.dot(np.dot(np.dot(inv1minZ, inv1minZ), Z), P[-1]))
    EQQmin1Total = 2 * sum(np.dot(np.dot(np.dot(np.dot(np.dot(inv1minZ, inv1minZ), inv1minZ), Z), Z), P[-1]))
    EQ2Total = EQQmin1Total + EQTotal

    # compute 1st and 2nd marginal moments of the numbers in the queue E[Q_i] and E[Q_i^2]
    EQ = alpha * EQTotal
    EQQmin1 = alpha * alpha * EQQmin1Total
    EQ2 = EQQmin1 + EQ

    # compute 1st and 2nd marginal moments of the numbers in the system E[N_i] and E[N_i^2]
    ENTotal = EQTotal + sum(lamda / mu)
    EN = EQ + lamda / mu

    # TODO compute the E[N_i^2]
    ES2 = np.zeros(mClasses)
    for (p, idx) in zip(P[:-1], idxMat[:-1]):
        ES2 += np.dot(p, idx ** 2)
    ES2 += np.dot(np.dot(inv1minZ, P[-1]), idxMat[-1] ** 2)

    ESq = alpha * np.dot(np.dot(np.dot(np.dot(inv1minZ, inv1minZ), Z), P[-1]), idxMat[-1])

    EN2 = EQ2 + 2 * ESq + ES2

    # compute marginal variances of the numbers in the queue Var[Q_i] and in the system Var[N_i]
    VarQTotal = EQ2Total - EQTotal ** 2
    VarQ = EQ2 - EQ ** 2

    VarN = EN2 - EN ** 2

    # computeMarginalDistributions
    qmax = 1500

    marginalN = np.zeros((mClasses, qmax))

    for m in xrange(mClasses):
        for imap, p in zip(i_map_full[:-1], P[:-1]):
            for i, idx in imap.items():
                marginalN[m, idx[m]] += p[i]

        inv1minAlphaZ = np.linalg.inv(np.eye(len(P[-1])) - (1 - alpha[m]) * Z)
        frac = np.dot(alpha[m] * Z, inv1minAlphaZ)
        # tmp = np.dot(self.Z, self.P[-1])
        # tmp = np.dot(inv1minAlphaZ, tmp)
        tmp = np.dot(inv1minAlphaZ, P[-1])

        for q in xrange(0, qmax):
            for i, idx in i_map_full[-1].items():
                if idx[m] + q < qmax: marginalN[m, idx[m] + q] += tmp[i]
            tmp = np.dot(frac, tmp)
    return marginalN, EN, VarN


def whittApprox(E1, E2, E3):
    '''
    input: first 3 moments of hyperexpo dist.
    returns: parameters of hyperexpo (p, v1 and v2)
    uses whitt approximation.....
    '''
    x = E1 * E3 - 1.5 * E2 ** 2
    # print x
    assert x >= 0.0

    y = E2 - 2 * (E1 ** 2)
    # print y
    assert y >= 0.0

    Ev1 = ((x + 1.5 * y ** 2 + 3 * E1 ** 2 * y) + math.sqrt(
        (x + 1.5 * y ** 2 - 3 * E1 ** 2 * y) ** 2 + 18 * (E1 ** 2) * (y ** 3))) / (6 * E1 * y)
    # print Ev1
    assert Ev1 >= 0

    Ev2 = ((x + 1.5 * y ** 2 + 3 * E1 ** 2 * y) - math.sqrt(
        (x + 1.5 * y ** 2 - 3 * E1 ** 2 * y) ** 2 + 18 * (E1 ** 2) * (y ** 3))) / (6 * E1 * y)
    assert Ev2 >= 0

    p = (E1 - Ev2) / (Ev1 - Ev2)
    assert p >= 0

    return 1.0 / Ev1, 1.0 / Ev2, p

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
        E_S1 = (np.inner(lamda, 1 / mu) - (lamda[mCl] * 1 / mu[mCl])) / (sum(lamda) - lamda[mCl])  # checked

        # second moment
        E_S2 = 2 * (np.inner(lamda, (1 / mu) ** 2) - (lamda[mCl] * (1 / mu[mCl]) ** 2)) / (
        sum(lamda) - lamda[mCl])  # checked

        # third moment
        E_S3 = 6 * (np.inner(lamda, (1 / mu) ** 3) - (lamda[mCl] * (1 / mu[mCl]) ** 3)) / (
        sum(lamda) - lamda[mCl])  # checked

        # calculate inputs for to check neccesity condtion:
        varA = E_S2 - E_S1 ** 2
        cA = math.sqrt(varA) / E_S1

        # to check if all of the service rates of approximated service are same
        # if it is true sum of hyperexpo with same parameter is ---> exponential distribution

        mu_copy = []
        mu_copy[:] = mu
        del mu_copy[mCl]

        if isServiceRateEqual(mu_copy) == True:
            # we can assume there is only aggreate remaing streams to one rather than two
            p = 1
            v1 = mu_copy[0]

            lam1 = lamda[mCl]
            # S1=1/mu[mCl]

            lamA1 = p * (sum(lamda) - lamda[mCl])
            # SA1=1/float(v1)

            # if sum(lamda/mu)>nservers:
            #    nservers
            # we have only two streams now so mClasses=2
            marginalLength, ENLength, VarLength = MMCsolver(np.array([lam1, lamA1]), \
                                                            np.array([mu[mCl], v1]), nservers=nServers, mClasses=2)

            marginalN.append(marginalLength[0])
            EN.append(ENLength[0])
            VarN.append(VarLength[0])

        # if (E_S3-(3.0/2.0)*((1+cA**2)**2)*E_S1**3)<0.0:
        # E_S3=(3.0/2.0)*((1+cA**2)**2)*E_S1**3+0.01
        #    print "aaa"
        #    v1, v2, p=whittApprox(E_S1, E_S2, E_S3)

        else:
            # a2 calculation
            a2 = (6 * E_S1 - (3 * E_S2 / E_S1)) / ((6 * E_S2 ** 2 / 4 * E_S1) - E_S3)

            # a1 calculation
            a1 = (1 / E_S1) + (a2 * E_S2 / (2 * E_S1))

            # v1 calculation
            v1 = (1.0 / 2.0) * (a1 + math.sqrt(a1 ** 2 - 4 * a2))

            # v2 calculation
            v2 = (1.0 / 2.0) * (a1 - math.sqrt(a1 ** 2 - 4 * a2))

            # p calculation
            p = 1 - ((v2 * (E_S1 * v1 - 1)) / float((v1 - v2)))

            lam1 = lamda[mCl]
            # S1=1/mu[mCl]

            lamA1 = p * (sum(lamda) - lamda[mCl])
            # SA1=1/float(v1)


            lamA2 = (1 - p) * (sum(lamda) - lamda[mCl])
            # SA2=1/float(v2)


            # Now we have 3 classes of streams (2 streams for approximation) as usual
            # so mClasses=3

            marginalLength, ENLength, VarLength = MMCsolver(np.array([lam1, lamA1, lamA2]), \
                                                            np.array([mu[mCl], v1, v2]), nservers=nServers, mClasses=3)

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
        E_S1 = (np.inner(lamda, 1 / mu) - (lamda[mCl] * 1 / mu[mCl])) / (sum(lamda) - lamda[mCl])  # checked
        # print E_S1
        # second moment
        E_S2 = 2 * (np.inner(lamda, (1 / mu) ** 2) - (lamda[mCl] * (1 / mu[mCl]) ** 2)) / (
        sum(lamda) - lamda[mCl])  # checked

        # third moment
        E_S3 = 6 * (np.inner(lamda, (1 / mu) ** 3) - (lamda[mCl] * (1 / mu[mCl]) ** 3)) / (
        sum(lamda) - lamda[mCl])  # checked

        # calculate inputs for to check neccesity condtion:
        varA = E_S2 - E_S1 ** 2
        cA = math.sqrt(varA) / E_S1

        assert (E_S3 - (3.0 / 2.0) * ((1 + cA ** 2) ** 2) * E_S1 ** 3) > 0

        # to check if all of the service rates of approximated service are same
        # if it is true sum of hyperexpo with same parameter is ---> exponential distribution

        mu_copy = []
        mu_copy[:] = mu
        del mu_copy[mCl]

        if isServiceRateEqual(mu_copy) == True:
            # we can assume there is only aggreate remaing streams to one rather than two
            p = 1
            v1 = mu_copy[0]

            lam1 = lamda[mCl]
            # S1=1/mu[mCl]

            lamA1 = p * (sum(lamda) - lamda[mCl])
            # SA1=1/float(v1)

            # sum(lamda/mu)<nservers

            if sum(np.array([lam1, lamA1]) / np.array([mu[mCl], v1])) > nservers:
                # print "hasan"
                nservers = int(sum(np.array([lam1, lamA1]) / np.array([mu[mCl], v1]))) + 1

            # we have only two streams now so mClasses=2
            marginalLength, ENLength, VarLength = MMCsolver(np.array([lam1, lamA1]), \
                                                            np.array([mu[mCl], v1]), nservers, mClasses=2)

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

            lamA1 = p * (sum(lamda) - lamda[mCl])
            # SA1=1/float(v1)


            lamA2 = (1 - p) * (sum(lamda) - lamda[mCl])
            # SA2=1/float(v2)

            if sum(np.array([lam1, lamA1, lamA2]) / np.array([mu[mCl], v1, v2])) >= nservers:
                # print "turan"
                nservers = int(sum(np.array([lam1, lamA1, lamA2]) / np.array([mu[mCl], v1, v2]))) + 1
            # Now we have 3 classes of streams (2 streams for approximation) as usual
            # so mClasses=3

            marginalLength, ENLength, VarLength = MMCsolver(np.array([lam1, lamA1, lamA2]), \
                                                            np.array([mu[mCl], v1, v2]), nservers, mClasses=3)

            marginalN.append(marginalLength[0])
            EN.append(ENLength[0])
            VarN.append(VarLength[0])

    return marginalN, EN, VarN, nservers


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
    EBO = np.sum(marginalDistribution * np.array(range(marginalDistribution.shape[1])), axis=1)

    hb_ratio = holdingCosts / penalty
    for sk in xrange(nSKUs):
        while S[sk] < maxQueue and np.sum(marginalDistribution[sk, S[sk] + 1:]) > hb_ratio[sk]:
            S[sk] += 1
            PBO[sk] = np.sum(marginalDistribution[sk, S[sk] + 1:])  # -= marginalDistribution[sk, S[sk]]
            EBO[sk] = np.sum(marginalDistribution[sk, S[sk]:] * n_array[:-S[sk]])  # -= PBO[sk]

    totalCost = np.sum(S * holdingCosts) + np.sum(penalty * EBO)
    hCost = np.sum(S * holdingCosts)
    pCost = np.sum(penalty * EBO)
    # print ((EBO < 0).sum() == EBO.size).astype(np.int)
    # if pCost<0.0:
    #    print EBO
    # print  ((EBO < 0).sum() == EBO.size).astype(np.int)
    # print all(i >= 0.0 for i in marginalDistribution)

    return totalCost, hCost, pCost, S, EBO


def Kmedian(serviceRates, NumberofCluster, holding_costs, similarityType=1):
    '''
    input: -service rates for each SKU
           -number of cluster: between 1,2,..., number of SKUs
           -holding costs for each SKU
           -similarity Type---> Service Rate (1)
                                Service Rates x holding costs (2)


    output: -Clusters and SKUs in each cluster under different similarirty measure (1) Service rate (2)cxmu rule

    Uses IP formulation and solver to find cluster.  Similarity is based on service rates.
    Pulp package has to be imported...

    '''

    CLUSTERS = NumberofCluster
    SKUs = range(1, len(serviceRates) + 1)

    # cost matrix set up
    if similarityType == 1:

        size = (len(serviceRates), len(serviceRates))
        Cij2 = np.zeros(size)
        for i in range(len(serviceRates)):
            for j in range(i + 1, len(serviceRates)):
                Cij2[i, j] = Cij2[j, i] = (serviceRates[i] - serviceRates[j]) ** 2
    else:

        size = (len(serviceRates), len(serviceRates))
        Cij2 = np.zeros(size)
        for i in range(len(serviceRates)):
            for j in range(i + 1, len(serviceRates)):
                Cij2[i, j] = Cij2[j, i] = (serviceRates[i] * holding_costs[i] - serviceRates[j] * holding_costs[j]) ** 2

    # problem name and objective type
    prob = pulp.LpProblem("Kmedian", LpMinimize)

    # Decision variables
    assign_vars = pulp.LpVariable.dicts("AtCluster", [(i, j) for i in SKUs for j in SKUs], 0, 1, LpBinary)
    use_vars = pulp.LpVariable.dicts("UseCluster", SKUs, 0, 1, LpBinary)

    # Objective function
    prob += pulp.lpSum(assign_vars[(i, j)] * Cij2[i - 1, j - 1] for i in SKUs for j in SKUs)

    # Constraints
    for i in SKUs:
        prob += pulp.lpSum(assign_vars[(i, j)] for j in SKUs) == 1

    prob += pulp.lpSum(use_vars[j] for j in SKUs) == CLUSTERS

    for j in SKUs:
        for i in SKUs:
            if j != i:
                prob += assign_vars[(i, j)] <= use_vars[j]
            else:
                prob += assign_vars[(i, j)] == use_vars[j]

    # solving and arranging results
    prob.solve()
    result = []

    for j in SKUs:
        if use_vars[j].varValue > 0:
            cluster = []
            for i in SKUs:
                if assign_vars[(i, j)].varValue > 0:
                    cluster.append(i)
            result.append(cluster)
    return result


def KmedianHeuristic(FailureRates, ServiceRates, holding_costs, penalty_cost, skillCost, machineCost, similarityType):
    '''
    input: -Failure rates and corresponding service rates of each SKU
           -Related cost terms holding costs for SKUs(array), backorder, skill and server (per server and per skill)
           -Kmedian function to cluster (group) SKUs based on service rates and cxmu rule -->to generate possible solutions
           -Kmedian needs--> pulp (import pulp)
           -MMCsolver and Approx_MMCsolver functions--> to find Queue length dist. of failed SKUs
                                                    --> number of SKUS >=4 use approximation

           -OptimizeStockLevels calculates EBO and S for giving clustering (Queue length dist.)

    output: Returns best total cost and other cost terms, Expected backorder (EBO) and stocks (S) for each SKU

    Kmedian algorithm finds the best clusters by solving and IP for given number of clusters. For varying number of
    clusters (1,2,..., #SKUs), Total costs are calculated and minimum of them is reported.

    Warning !! type matching array vs list might be problem (be careful about type matching)

    '''
    bestCost = float('inf')
    bestCluster = []
    bestS = []
    bestEBO = []
    bestserverAssignment = []
    sliceIndex2 = []
    LogFileList = []
    logFile = {}
    iterationNum = 0
    for NumberofCluster in range(1, len(FailureRates) + 1):
        cluster_KMedian = Kmedian(ServiceRates, NumberofCluster, holding_costs, similarityType)
        TotalCost = 0.0
        TotalHolding, TotalPenalty, TotalSkillCost, TotalMachineCost = 0.0, 0.0, 0.0, 0.0
        EBO_cluster = []
        S_cluster = []
        serverAssignment = []
        for cluster in cluster_KMedian:
            sliceIndex2[:] = cluster
            sliceIndex2[:] = [x - 1 for x in sliceIndex2]

            sRate = np.array(ServiceRates[sliceIndex2])
            fRate = np.array(FailureRates[sliceIndex2])
            hcost = np.array(holding_costs[sliceIndex2])

            min_nserver = int(sum(fRate / sRate)) + 1

            # costTemp=0
            # while costTemp<=machineCost:
            if len(sRate) <= 3:
                marginalDist, EN, VarN = MMCsolver(fRate, sRate, min_nserver, len(fRate))
            else:
                marginalDist, EN, VarN, min_nserverUpdate = Approx_MMCsolver2(fRate, sRate, min_nserver, len(fRate))
                min_nserver = min_nserverUpdate

            totalCostClust, hCost, pCost, S, EBO = OptimizeStockLevelsAndCosts(hcost, penalty_cost,
                                                                               np.array(marginalDist))

            # increasing number of servers and checking if total cost decreases

            TotalMachine_Cost = min_nserver * machineCost
            TotalSkill_Cost = min_nserver * len(fRate) * skillCost

            totalCostClust = totalCostClust + TotalMachine_Cost + TotalSkill_Cost
            # temp_totalCostClust=0.0
            # print totalCostClust
            # print min_nserver
            # print "**********"
            while True:
                min_nserver += 1
                if len(sRate) <= 3:
                    marginalDist, EN, VarN = MMCsolver(fRate, sRate, min_nserver, len(fRate))
                else:
                    marginalDist, EN, VarN, min_nserverUpdate = Approx_MMCsolver2(fRate, sRate, min_nserver, len(fRate))
                    min_nserver = min_nserverUpdate

                temp_totalCostClust, temp_hCost, temp_pCost, temp_S, temp_EBO = OptimizeStockLevelsAndCosts(hcost,
                                                                                                            penalty_cost,
                                                                                                            np.array(
                                                                                                                marginalDist))
                temp_TotalMachine_Cost = min_nserver * machineCost
                temp_TotalSkill_Cost = min_nserver * len(fRate) * skillCost

                temp_totalCostClust = temp_totalCostClust + temp_TotalMachine_Cost + temp_TotalSkill_Cost

                if temp_totalCostClust > totalCostClust:
                    min_nserver -= 1
                    break
                else:
                    totalCostClust = temp_totalCostClust
                    # print totalCostClust
                    # print min_nserver
                    # print "=========="
                    TotalMachine_Cost = temp_TotalMachine_Cost
                    TotalSkill_Cost = temp_TotalSkill_Cost
                    hCost = temp_hCost
                    pCost = temp_pCost
                    S = temp_S
                    EBO = temp_EBO

            TotalHolding += hCost
            TotalPenalty += pCost

            TotalSkillCost += TotalSkill_Cost
            TotalMachineCost += TotalMachine_Cost

            TotalCost = TotalCost + totalCostClust

            EBO_cluster.append(EBO.tolist())
            S_cluster.append(S.tolist())
            serverAssignment.append(min_nserver)

        iterationNum += 1
        # Log file preparation
        logFile["iterationNum"] = iterationNum
        logFile["TotalCost"] = TotalCost
        logFile["TotalMachineCost"] = TotalMachineCost
        logFile["TotalSkillCost"] = TotalSkillCost
        logFile["TotalHolding"] = TotalHolding
        logFile["TotalPenalty"] = TotalPenalty
        logFile["S"] = S_cluster
        logFile["EBO"] = EBO_cluster
        logFile["Assignment"] = cluster_KMedian
        logFile["serverAssignment"] = serverAssignment

        LogFileList.append(logFile)

        logFile = {}

        if TotalCost < bestCost:
            bestCost = TotalCost
            bestHolding = TotalHolding
            bestPenalty = TotalPenalty
            bestMachineCost = TotalMachineCost
            bestSkillCost = TotalSkillCost
            bestCluster = cluster_KMedian[:]
            bestS[:] = S_cluster
            bestEBO[:] = EBO_cluster
            bestserverAssignment[:] = serverAssignment

    return bestCost, bestHolding, bestPenalty, bestMachineCost, bestSkillCost, bestCluster, bestS, bestEBO, \
           bestserverAssignment, LogFileList

json_case=[]
with open("fullRangeResultsFullFlexNew.json", "r") as json_file:
    #json_file.readline()
    for line in json_file:
        json_case.append(json.loads(line))

results = []
KmedianResult = {}

for case in json_case:
    if case["caseID"] != "Case: 000x":
        failure_rates = case["failure_rates"]
        service_rates = case["service_rates"]
        holding_costs = case["holding_costs"]
        skill_cost = case["skill_cost"]
        penalty_cost = case["penalty_cost"]
        machine_cost = case["machine_cost"]

        # print "solving: "+str(case["caseID"])


        for similarityType in [1, 2]:
            start_time = time.time()

            bestCost, bestHolding, bestPenalty, bestMachineCost, bestSkillCost, bestCluster, bestS, bestEBO, \
            bestserverAssignment, LogFileList = KmedianHeuristic(np.array(failure_rates), np.array(service_rates), \
                                                                 np.array(holding_costs), penalty_cost, skill_cost,
                                                                 machine_cost, similarityType)
            stop_time = time.time() - start_time

            print bestCost
            KmedianResult["caseID"] = case["caseID"]

            KmedianResult["Kmedianruntime"] = stop_time
            KmedianResult["KmedianTotalCost"] = bestCost
            KmedianResult["KmedianHoldingCost"] = bestHolding
            KmedianResult["KmedianPenaltyCost"] = bestPenalty
            KmedianResult["KmedianMachineCost"] = bestMachineCost
            KmedianResult["KmedianSkillCost"] = bestSkillCost

            KmedianResult["KmedianCluster"] = bestCluster
            KmedianResult["KmedianS"] = bestS
            KmedianResult["KmedianEBO"] = bestEBO
            KmedianResult["KmedianserverAssignment"] = bestserverAssignment
            KmedianResult["KmedianLogFile"] = LogFileList

            KmedianResult["GAP"] = bestCost - case["total_cost"]
            KmedianResult["PercentGAP"] = 100 * (bestCost - case["total_cost"]) / case["total_cost"]
            KmedianResult["similarityType"] = str(similarityType)
            KmedianResult["GAresults"] = case

            results.append(KmedianResult)

            KmedianResult = {}

with open('KmedianAll.json', 'w') as outfile:
    json.dump(results, outfile)