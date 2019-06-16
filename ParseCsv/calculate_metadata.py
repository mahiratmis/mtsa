import ast
import pandas as pd
import csv

# calculate following metadata
header = ['num_clusters', 'num_used_servers', 'avg_num_servers_per_cluster', 'avg_num_skills_per_server', 'avg_cross_training_perf', 'total_stock']

# Open the CSV
df = pd.read_csv('../results/results_MSSA_Addaptive_all.csv')
#df = pd.read_csv('../results/results_CIE_all.csv')
#df = pd.read_csv('../results/results_MSSA_all.csv')
print (df.info())
#write new statistics to meta_stats.csv file
with open("meta_stats.csv", 'wb') as resultFile:
    wr = csv.writer(resultFile, dialect='excel')

    wr.writerow(header)
    #iterate over the original results
    for index, row in df.iterrows():
        #convert raw string to python lists
        best_server_assignment = ast.literal_eval(row['bestServerAssignment'])
        best_cluster = ast.literal_eval(row['best_cluster'])
        best_s = ast.literal_eval(row['bestS'])
        #get meta results
        num_clusters = len(best_server_assignment)
        num_used_servers = sum(best_server_assignment)
        avg_num_servers_per_cluster = 1.0 * num_used_servers / num_clusters
        avg_num_skills_per_server = sum([len(x[0]) * x[1] for x in zip(best_cluster, best_server_assignment)]) / (1.0 * num_used_servers)
        avg_cross_training_perf = 100 * avg_num_skills_per_server / sum([len(x) for x in best_cluster])
        total_stock = sum([sum(x) for x in best_s])
        wr.writerow([num_clusters, num_used_servers, avg_num_servers_per_cluster, avg_num_skills_per_server, avg_cross_training_perf, total_stock])
