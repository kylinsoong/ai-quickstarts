import matplotlib.pyplot as plt

x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 ,14]
y = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196]

plt.plot(x, y)

plt.title('y = x^2')
plt.xlabel('X')
plt.ylabel('Y')

plt.show()
