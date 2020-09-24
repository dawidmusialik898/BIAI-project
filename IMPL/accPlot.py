import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

x = [6, 9, 12]
y = [66.40, 71.57, 73.04]
pyplot.plot(x, y, color='green')

pyplot.savefig("accuracy" + '_plot.png')
pyplot.close()