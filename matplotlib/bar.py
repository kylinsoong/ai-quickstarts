import matplotlib.pyplot as plt

categories = ['A', 'B', 'C', 'D']
values = [4, 7, 1, 8]

plt.bar(categories, values)

plt.title('Simple Bar Plot')
plt.xlabel('Categories')
plt.ylabel('Values')

plt.show()

