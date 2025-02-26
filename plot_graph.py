import os
import plotly
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line

import numpy as np
from pprint import pprint

class Plot_Graph_Data:
    def __init__(self, path, title, metrics):
        self.path = path
        self.title = title
        self.metrics = metrics

    def __repr__(self):
        return 'plot_graph.Plot_Graph_Data(\'{}\',\'{}\', {})'.format(self.path, self.title, self.metrics)

class Plot_Graph:

    def __init__(self, plotGraphDatas):
        self.counts = {}
        self.pathes = {}
        self.datas = {}

        for data in plotGraphDatas:
            self.counts[data.title] = 0
            self.pathes[data.title] = data.path
            self.datas[data.title] = data.metrics

            if not os.path.exists(data.path):
                os.makedirs(data.path)

    def addDatas(self, title, keys, datas):
        for key, data in zip(keys, datas):
            self.datas[title][key].append(data)
        self.counts[title] += 1

    def plot(self, title, meanFlag=True):
        plot_datas = []

        for key in self.datas[title].keys():
            data = self.datas[title][key]

            # plot_datas_x += [np.arange(len(data))]
            # plot_means_x += [x[::self.counts[title]]]
            
            # plot_datas += [np.array(data)]
            # plot_means += [data.reshape(-1, self.counts[title]).mean(axis=1)]

            x  = np.arange(len(data))
            data = np.array(data)
            plot_datas.append(Scatter(x=x,  y=data, name=key,           mode="lines"))
            
            if meanFlag:
                x2 = x.reshape(-1, self.counts[title]).mean(axis=1)
                mean = data.reshape(-1, self.counts[title]).mean(axis=1)
                plot_datas.append(Scatter(x=x2, y=mean, name=key + '_mean', mode="lines"))

        self.counts[title] = 0

        plotly.offline.plot(plot_datas, filename=os.path.join(self.pathes[title], title + '.html'), auto_open=False)

    
if __name__ == '__main__':
    data1 = Plot_Graph_Data('./plot/data1', 'abplot', {'a': [], 'b': []})
    data2 = Plot_Graph_Data('./plot/data2', 'cplot',  {'c': []}         )

    plotGraph = Plot_Graph([data1, data2])

    plotGraph.addDatas('abplot', ['a', 'b'], [0, 10])

    for i in range(10):
        for j in range(10):
            plotGraph.addDatas('cplot', ['c'], [10*j+i])
        plotGraph.plot('cplot')

