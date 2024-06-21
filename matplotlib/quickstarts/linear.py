import matplotlib.pyplot as plt
import tensorflow as tf

x = tf.constant(range(15), dtype=tf.float32)
y = 2 * x + 10

x = [0, 1, 2, 3, 4, 5, 6 ,7 ,8, 9, 10, 11, 12, 13 ,14]
y = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38]

plt.plot(x, y, color='green', linestyle='dashed', marker='o')

plt.title('Linear: y = 2x + 10')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)

plt.show()
