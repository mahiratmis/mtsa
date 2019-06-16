
import csv
import json

import plotly.plotly as py
import pandas as pd
from plotly.graph_objs import *

py.sign_in('atmismahir', '****')

# Open the CSV  
df = pd.read_csv('./mssainner_json.csv')

# print data.info()

columns = list(df.columns)
skills = list(df["skill"])
del df["skill"]
data = df.values.tolist()
min_val, max_val = df.values.min(), df.values.max()
print min_val, max_val


traces = []
for skill_name in df.columns:
    trace = {
        "y": list(df[skill_name]),
        "boxpoints": "all",
        "fillcolor": "rgb(255, 255, 255)",
        "jitter": 0.4,
        "line": {"width": 1},
        "marker": {
            "line": {"width": 0},
            "opacity": 0.9,
            "size": 2
        },
        "name": skill_name,
        "opacity": 0.99,
        "type": "box"
    }
    traces.append(trace)


data = Data(traces)

layout = {
    "autosize": False,
    "bargap": 0.2,
    "bargroupgap": 0,
    "barmode": "stack",
    "boxgap": 0.2,
    "boxgroupgap": 0.3,
    "boxmode": "overlay",
    "dragmode": "zoom",
    "font": {
        "color": "#000",
        "family": "Arial, sans-serif",
        "size": 12
    },
    "height": 450,
    "hovermode": "x",
    "legend": {
        "bgcolor": "#fff",
        "bordercolor": "#000",
        "borderwidth": 1,
        "font": {
            "color": "chocolate",
            "family": "Arial, sans-serif",
            "size": 1
        },
        "traceorder": "normal"
    },
    "margin": {
        "r": 80,
        "t": 80,
        "b": 140,
        "l": 80,
        "pad": 2
    },
    "paper_bgcolor": "rgb(255, 255, 255)",
    "plot_bgcolor": "rgb(255, 255, 255)",
    "showlegend": False,
    "title": "MSSA_Inner Results Compared to Flexible and Dedicated",
    "titlefont": {
        "color": "chocolate",
        "family": "Arial, sans-serif",
        "size": 12
    },
    "width": 1100,
    "xaxis": {
        "autorange": False,
        #"autotick": True,
        "dtick": 1,
        "exponentformat": "e",
        "gridcolor": "#ddd",
        "gridwidth": 1,
        "linecolor": "rgb(255, 255, 255)",
        "linewidth": 0.1,
        "mirror": True,
        "nticks": 0,
        "range": [-1.44864344477, 50.3540302404],
        "showexponent": "all",
        "showgrid": False,
        "showticklabels": True,
        "tick0": 0,
        "tickangle": 90,
        "tickcolor": "#000",
        "tickfont": {
            "color": "chocolate",
            "family": "Arial, sans-serif",
            "size": 12
        },
        "ticklen": 5,
        "ticks": "",
        "tickwidth": 1,
        "title": "Skill Names",
        "titlefont": {
            "color": "chocolate",
            "family": "Arial, sans-serif",
            "size": 1
        },
        "type": "category",
        "zeroline": False,
        "zerolinecolor": "#000",
        "zerolinewidth": 1
    },
    "yaxis": {
        "autorange": False,
        # "autotick": True,
        "dtick": 10,
        "exponentformat": "e",
        "gridcolor": "white",
        "gridwidth": 1,
        "linecolor": "rgb(255, 255, 255)",
        "linewidth": 0.1,
        "mirror": True,
        "nticks": 0,
        "range": [-1.0, 320],
        "showexponent": "all",
        "showgrid": False,
        "showticklabels": True,
        "tick0": 0,
        #"tickangle": "auto",
        "tickcolor": "#000",
        "tickfont": {
            "color": "chocolate",
            "family": "Arial, sans-serif",
            "size": 1
        },
        "ticklen": 5,
        "ticks": "",
        "tickwidth": 1,
        "title": "Total Costs",
        "titlefont": {
            "color": "chocolate",
            "family": "Arial, sans-serif",
            "size": 12
        },
        "type": "linear",
        "zeroline": False,
        "zerolinecolor": "#000",
        "zerolinewidth": 1
    }
}
fig = Figure(data=data, layout=layout)
plot_url = py.plot(fig)
