from matplotlib import pyplot as plt
import numpy as np
x = np.linspace(0, np.pi)
y_sin = np.sin(x)
y_cos = np.cos(x)
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(224)
ax1.plot(x, y_sin)
ax2.plot(x, y_sin, 'go--', linewidth=2, markersize=12)
ax3.plot(x, y_cos, color='red', marker='+', linestyle='dashed')
plt.show()