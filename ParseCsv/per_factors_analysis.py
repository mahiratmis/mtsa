import ast
import pandas as pd
import csv
import itertools

# Open the CSV
df = pd.read_csv('../results/all_results_with_meta.csv')
# print df.info()

algorithm_types = df["algorithm_type"].unique()

# slice first 128 rows, and columns from n_sku to cost_ci
factors_df = df.loc[:127, 'n_sku':'cost_ci']
factor_names = factors_df.columns.tolist()



run_time_avgs = {}
benchmark_name_factor_mean = []
for alg_type in algorithm_types:
    if alg_type in ["MSSA_Adaptive", "MSSA_Inner", "MSSA", "CIE_SA"]:
        # print (df[df["algorithm_type"] == alg_type])
        # print (df[df["algorithm_type"] == alg_type].mean(skipna=True))
        # run_time_avgs[alg_type] = round(df[df["algorithm_type"] == alg_type].mean(skipna=True)["running_time"], 2)
        means_per_factor = {}
        for factor_name in factor_names:
            factor_name_unique_vals = df[factor_name].unique()
            for val in factor_name_unique_vals:
                tc_mean = df[df[factor_name] == val][alg_type].mean()
                if not isinstance(val, str) and val < 1:
                    val = int(val * 100)  # handle decimal values
                means_per_factor[factor_name + "_" + str(val)] = tc_mean
        factor_df = pd.DataFrame(means_per_factor, index=[benchmark_name])
        benchmark_name_factor_mean.append(factor_df)
with open('runtime_means.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in run_time_avgs.items():
        writer.writerow([key, value])
exit(0)

print (algorithm_types)

# get binary combinations of algorithms
combinations = itertools.combinations(algorithm_types, r=2)

# calculate total cost difference per algorithm pair
total_cost_benchmarks = {}

# stats_col = "total_cost"

# for alg1, alg2 in combinations:
#     print  alg1,alg2
#     tc_first = df[df["algorithm_type"] == alg1][stats_col].values  # to numpy
#     tc_second = df[df["algorithm_type"] == alg2][stats_col].values
#     total_cost_percentage = (tc_first - tc_second) * 100.0 / tc_first
#     total_cost_benchmarks[alg1 + "_" + alg2] = total_cost_percentage

# # columns benchmark names, rows total cost % performance
# total_cost_diffs_df = pd.DataFrame(total_cost_benchmarks)
# # save intermediate result
# total_cost_diffs_df.to_csv("total_cost_diffs_df.csv")

# # get the names of the compared algorithms
# benchmark_names = total_cost_diffs_df.columns.tolist()

# # slice first 128 rows, and columns from n_sku to cost_ci
# factors_df = df.loc[:127, 'n_sku':'cost_ci']
# factor_names = factors_df.columns.tolist()

# result_df = pd.concat([total_cost_diffs_df, factors_df], axis=1, sort=False)

# benchmark_name_factor_mean = []
# for benchmark_name in benchmark_names:
#     means_per_factor = {}
#     for factor_name in factor_names:
#         factor_name_unique_vals = result_df[factor_name].unique()
#         for val in factor_name_unique_vals:
#             tc_mean = result_df[result_df[factor_name] == val][benchmark_name].mean()
#             if not isinstance(val, str) and val < 1:
#                 val = int(val*100)  # handle decimal values
#             means_per_factor[factor_name + "_" + str(val)] = tc_mean
#     factor_df = pd.DataFrame(means_per_factor, index=[benchmark_name])
#     benchmark_name_factor_mean.append(factor_df)

# final_result_df = pd.concat(benchmark_name_factor_mean, sort=False)
# print final_result_df.info()
# print final_result_df.index
# print final_result_df.columns

# # save new statistics as a csv file
# final_result_df.to_csv("tc_benchmark.csv")



stats_col = "running_time"

for alg1, alg2 in combinations:
    tc_first = df[df["algorithm_type"] == alg1][stats_col].values  # to numpy
    tc_second = df[df["algorithm_type"] == alg2][stats_col].values
    total_cost_percentage = (tc_first - tc_second) * 100.0 / tc_first
    total_cost_benchmarks[alg1 + "_" + alg2] = total_cost_percentage

# columns benchmark names, rows total cost % performance
total_cost_diffs_df = pd.DataFrame(total_cost_benchmarks)
# save intermediate result
total_cost_diffs_df.to_csv(stats_col + "_diffs_df.csv")

# get the names of the compared algorithms
benchmark_names = total_cost_diffs_df.columns.tolist()



result_df = pd.concat([total_cost_diffs_df, factors_df], axis=1, sort=False)

benchmark_name_factor_mean = []
for benchmark_name in benchmark_names:
    means_per_factor = {}
    for factor_name in factor_names:
        factor_name_unique_vals = result_df[factor_name].unique()
        for val in factor_name_unique_vals:
            tc_mean = result_df[result_df[factor_name] == val][benchmark_name].mean()
            if not isinstance(val, str) and val < 1:
                val = int(val*100)  # handle decimal values
            means_per_factor[factor_name + "_" + str(val)] = tc_mean
    factor_df = pd.DataFrame(means_per_factor, index=[benchmark_name])
    benchmark_name_factor_mean.append(factor_df)

final_result_df = pd.concat(benchmark_name_factor_mean, sort=False)
print (final_result_df.info())
print (final_result_df.index)
print (final_result_df.columns)

# save new statistics as a csv file
final_result_df.to_csv(stats_col + "_benchmark.csv") 
