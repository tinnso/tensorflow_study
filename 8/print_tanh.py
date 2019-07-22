import numpy as np
import matplotlib.pyplot as plt
#x = np.linspace(-100, 100, 1000)
#y = np.tanh(x)

x = np.linspace(-3.14, 3.14, 1000)
y = np.sin(x)

plt.plot(x, y, label = "label", color = "red", linewidth = 2)
plt.xlabel("abscissa")
plt.ylabel("ordinate")
plt.title("tanh Example")
plt.show()