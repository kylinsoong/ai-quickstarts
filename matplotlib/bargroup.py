import matplotlib.pyplot as plt
import numpy as np

labels = ['Group 1', 'Group 2', 'Group 3', 'Group 4']
group1 = [20, 34, 30, 35]
group2 = [25, 32, 34, 20]
group3 = [30, 25, 24, 32]

x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, group1, width, label='Group 1')
rects2 = ax.bar(x, group2, width, label='Group 2')
rects3 = ax.bar(x + width, group3, width, label='Group 3')

ax.set_xlabel('Groups')
ax.set_ylabel('Scores')
ax.set_title('Scores by group and category')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(rects1)
add_labels(rects2)
add_labels(rects3)

plt.show()

