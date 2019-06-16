
import csv  
import json

import plotly.plotly as py
import pandas as pd
import plotly.graph_objs as go



py.sign_in('atmismahir', '****')
  
# Open the CSV  
df = pd.read_csv('../results/tc_benchmark.csv')
# df = pd.read_csv('../results/results_running_time_benchmark.csv')
#print df.info()

benchmarks = list(df["benchmark_types"])
del df["benchmark_types"]
columns = list(df.columns)

vals = df.values.tolist()
min_val, max_val = df.values.min(), df.values.max()
trace = go.Heatmap(z=vals,
                   x=columns,
                   y=benchmarks
)
data = [trace]

layout = go.Layout(
        title='Total Cost Benchmark',
        xaxis=dict(title='<b>Factors</b>', ticks=''),
        yaxis=dict(title='<b>Compared Algorithms</b>', ticks='', type="category"),
        height=600,
        width=800,
        margin=dict(
            r=10,
            t=50,
            b=80,
            l=180
        )
)

fig = go.Figure(data=data, layout=layout)
plot_url = py.plot(fig)
