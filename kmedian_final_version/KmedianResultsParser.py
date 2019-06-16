import json
import operator

# print cases[0].keys()
# #This is the best solution found by KMedian for caseID 0
# print cases[0]['KmedianCluster']
# #This is the best solution found by KMedian for caseID 0
# print cases[1]['KmedianCluster']
#
# #Initalize MSA with Log file solutions
# print cases[0]['KmedianLogFile'][0].keys()
#
# print type(cases[0]['KmedianLogFile'][0])
# #All iterations solution for KMedian from 1 cluster to N cluster
# #Chose TOP 5 based on minimu total cost
# for it in cases[0]['KmedianLogFile']:
#     print it["Assignment"], it["TotalCost"]

# get n minimum cost assignments
def get_top_n_assignment(log_file_i, n):
    # sort clusters ascendingly by TotalCost
    sorted_assignments = sorted(log_file_i, key=operator.itemgetter('TotalCost'))
    # get topn Assignment
    return  [it['Assignment'] for it in sorted_assignments[:n]]

# Cluster to SA vector needed
def clusterTovector(assigment):
    vector_len = 0
    for cluster in assigment:
        vector_len += len(cluster)

    SA_vector = vector_len * [0]
    ID = 1
    for cluster in assigment:
        for sku in cluster:
            SA_vector[sku - 1] = ID
        ID += 1

    return SA_vector

def get_n_best_individuals(num_cases, n=5):
    with open('KMedian_FullDataSet/KmedianAll_holding.json', 'r') as infile:
        cases = json.load(infile)
    best_n_individuals = []
    for i in range(num_cases):
        topn_assignments = get_top_n_assignment(cases[i]['KmedianLogFile'],n)
        Sa_vector_list = [clusterTovector(assigment) for assigment in topn_assignments]
        best_n_individuals.append(Sa_vector_list)
    return best_n_individuals




