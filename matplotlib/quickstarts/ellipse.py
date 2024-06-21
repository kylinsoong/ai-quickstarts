import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

fig, ax = plt.subplots()

ellipse = Ellipse(xy=(0.5, 0.5), width=0.6, height=0.4, angle=30, edgecolor='b', facecolor='none')

ax.add_patch(ellipse)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax.grid(True)

ax.set_title('Ellipse Example')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')

plt.show()

