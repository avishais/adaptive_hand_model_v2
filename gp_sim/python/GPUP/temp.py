import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import animation

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
e1 = Ellipse(xy=(0.5, 0.5), width=0.5, height=0.2, angle=60, animated=True, edgecolor='m', linewidth=2., fill=False)
e2 = Ellipse(xy=(0.8, 0.8), width=0.5, height=0.2, angle=100, animated=True)
ax.add_patch(e1)
ax.add_patch(e2)

def init():
    return e1, e2,

def animate(i):
    e2.center = (0.5+i*0.001, 0.5)
    e2.angle = e2.angle + 0.5
    return e1, e2,

anim = animation.FuncAnimation(fig, animate, init_func=init, interval=1, blit=True)
plt.show()