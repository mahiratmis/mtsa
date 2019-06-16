import csv
import json

import plotly.plotly as py
import pandas as pd
import plotly.graph_objs as go
import numpy as np

py.sign_in('atmismahir', '****')

# Open the CSV
df = pd.read_csv('/home/atmis/Documents/Projects/Python/discreteSWarm_stats/results/all_results_with_meta.csv')
print df.info()
#exit(0)
columns = list(df.columns)
#skills = list(df["skill"])
#del df["skill"]
#data = df.values.tolist()
#min_val, max_val = df.values.min(), df.values.max()
#print min_val, max_val

# df = pd.read_csv("https://raw.githubusercontent.com/bcdunbar/datasets/master/iris.csv")

algorithm_names = df[columns[0]].unique().tolist()
num_rows = len(df.index)
list_indices = [i for i in range(len(algorithm_names))]
repeated_indices = np.repeat(list_indices, num_rows/len(algorithm_names))
algorithms = dict(
                            range=[0, len(algorithm_names)-1],
                            ticktext=algorithm_names,
                            tickvals=list_indices,
                            label="algorithm",
                            values=repeated_indices)


show = ['running_time', 'total_cost', 'avg_cross_training_perf', 'avg_num_skills_per_server', 'avg_num_servers_per_cluster', 'num_used_servers', 'num_clusters']
df = df[show]
dimensions = list([dict(
                            range=[df[col_name].min(), df[col_name].max()],
                            tickvals=df[col_name].unique().tolist(),
                            label=col_name,
                            values=df[col_name],
                            tickformat=".3f") for col_name in show])


dimensions.insert(0, algorithms)

data = [
    go.Parcoords(
        line=dict(color=repeated_indices,
                  colorscale='Jet',
                  showscale=True,
                  reversescale=False,
                  cmin=len(algorithm_names),
                  cmax=0),
        dimensions=dimensions
    )
]

layout = go.Layout(
    plot_bgcolor='#E5E5E5',
    paper_bgcolor='white',
    font=dict(size=11)
)

fig = go.Figure(data=data, layout=layout)
plot_url = py.plot(fig)
