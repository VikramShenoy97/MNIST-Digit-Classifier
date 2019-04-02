import os
import pandas as pd
import math
import numpy as np
from PIL import Image

from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go

def drawGraph(confusion_matrix, labels):
    filename = "history.csv"
    history = pd.read_csv(filename, header=0, low_memory=False)
    history_array = history.values
    epochs = history_array[:, 0]
    training_accuracy = history_array[:, 1]
    training_loss = history_array[:, 2]
    val_accuracy = history_array[:, 3]
    val_loss = history_array[:, 4]

    py.sign_in('VikramShenoy','x1Un4yD3HDRT838vRkFA')


    trace0 = go.Scatter(
    x = epochs,
    y = training_accuracy,
    mode = "lines",
    name = "Training Accuracy"
    )

    trace1 = go.Scatter(
    x = epochs,
    y = val_accuracy,
    mode = "lines",
    name = "Validation Accuracy"
    )
    data = go.Data([trace0, trace1])
    layout = go.Layout()
    fig = go.Figure(data=data, layout=layout)
    fig['layout']['xaxis'].update(title="Number of Epochs", range = [min(epochs), max(epochs)], dtick=len(epochs)/10, showline = True, zeroline=True,  mirror='ticks', linecolor='#636363', linewidth=2)
    fig['layout']['yaxis'].update(title="Accuracy", range = [0, 1], dtick=0.1, showline = True, zeroline=True, mirror='ticks',linecolor='#636363',linewidth=2)
    py.image.save_as(fig, filename="Accuracy_Graph.png")

    print "Accuracy Graph Created"


    trace0 = go.Scatter(
    x = epochs,
    y = training_loss,
    mode = "lines",
    name = "Training Loss"
    )

    trace1 = go.Scatter(
    x = epochs,
    y = val_loss,
    mode = "lines",
    name = "Validation Loss"
    )
    data = go.Data([trace0, trace1])
    layout = go.Layout()
    fig = go.Figure(data=data, layout=layout)
    fig['layout']['xaxis'].update(title="Number of Epochs", range = [min(epochs), max(epochs)], dtick=len(epochs)/10, showline = True, zeroline=True,  mirror='ticks', linecolor='#636363', linewidth=2)
    fig['layout']['yaxis'].update(title="Loss", dtick=0.1, showline = True, zeroline=True, mirror='ticks',linecolor='#636363',linewidth=2)
    py.image.save_as(fig, filename="Loss_Graph.png")
    print "Loss Graph Created"

    trace = go.Heatmap(z=confusion_matrix, x=labels, y=labels, reversescale=False, colorscale='Viridis')
    data=[trace]
    layout = go.Layout(
    title='Confusion Matrix',
    showlegend = True,
    xaxis = dict(dtick=1),
    yaxis = dict(dtick=1))
    fig = go.Figure(data=data, layout=layout)
    py.image.save_as(fig, filename="Confusion_Matrix.png")
    print "Confusion Matrix Created"
    return
