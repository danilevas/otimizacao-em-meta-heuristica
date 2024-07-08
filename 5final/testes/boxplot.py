import matplotlib.pyplot as plt
import numpy as np

# Sample data
data = [np.random.normal(0, std, 100) for std in range(1, 4)]
print(data)
labels = ['Set 1', 'Set 2', 'Set 3']

plt.boxplot(data)

plt.title('Boxplot Example')
plt.xlabel('Category')
plt.ylabel('Values')
plt.xticks([1, 2, 3], labels)

# Showing the plot
plt.show()