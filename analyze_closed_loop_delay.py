import matplotlib
matplotlib.use("qt4agg")
import pylab as pl
import numpy as np
import my_figure as myfig
from pathlib import Path


data = np.load("/Users/arminbahl/Desktop/trigger_results_no_vsync.npy")
root_path = Path("/Users/arminbahl/Desktop/preprocessed data/maxwell_paper")

t = data[:, 0]

d0 = []
d1 = []
d2 = []

for period in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]:

    i = np.where((data[:, 0] > period-0.2) & (data[:, 1] == 0))[0][0]

    d0.append(data[i - 10:i + 40, 0] - data[i, 0])
    d1.append(data[i - 10:i + 40, 1])
    d2.append(data[i - 10:i + 40, 2])

d0 = np.mean(d0, axis=0)
d1 = np.mean(d1, axis=0)
d2 = np.mean(d2, axis=0)

d1 -= d1.min()
d2 -= d2.min()

d1 /= d1.max()
d2 /= d2.max()

fig = myfig.Figure(title="Closed-loop delay")

p0 = myfig.Plot(fig, xpos=4, ypos=15, plot_height=3, plot_width=4, lw=1, title='Closed-loop delay',
                xl="Time (s)", xmin=-0.1, xmax=0.4, xticks=[0, 0.1, 0.2, 0.3, 0.4],
                yl="Brightness (a.u)", ymin=-0.1, ymax=1.1, yticks=[0, 0.5, 1])

myfig.Line(p0, x=d0, y=d1, lc='C0', zorder=2, lw=1, alpha=0.9, label='Set by computer')
myfig.Line(p0, x=d0, y=d2, lc='C1', zorder=2, lw=1, alpha=0.9, label='Measured by camera')


fig.savepdf(root_path / f"closed_loop_delay", open_pdf=True)



