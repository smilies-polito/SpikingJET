import os
import numpy as np
import matplotlib.pyplot as plt

def plot_stacked_histogram(data):
    categories = ['No', 'SDC', 'Yes']
    num_categories = len(categories)
    
    # Transpose the data to have each column representing a category
    data_transposed = np.array(data).T
    bottom = np.zeros(len(data))
    
    # Grouping the data into three sets for each category
    group_data = [data_transposed[i:i+3] for i in range(0, len(data_transposed), 3)]
    
    for i, group in enumerate(group_data):
        for j, cat_data in enumerate(group):
            plt.bar(range(len(data)), cat_data, bottom=bottom, label=categories[j] if i == 0 else "", alpha=0.7)
            bottom += cat_data
        bottom += 0.05  # Adding space between groups
    
    plt.xlabel('Column')
    plt.ylabel('Percentage')
    plt.title('Stacked Histogram')
    plt.legend()
    plt.xticks(range(len(data)), labels=[i if i%4!=0 else '' for i in range(1,len(data)+1)])
    plt.text(0.7, -10, 'ciao')
    plt.savefig(os.path.join(os.path.dirname(__file__), 'stacked_histogram.png'))
    plt.show()

# Example data: Each column represents a category, and each row represents a column
data = [
    [30, 20, 50],  # A
    [20, 40, 40],  # B
    [50, 40, 10],  # C
    [0,0,0],
    [30, 20, 50],  # A
    [20, 40, 40],  # B
    [50, 40, 10],  # C
    [0,0,0],
    [30, 20, 50],  # A
    [20, 40, 40],  # B
    [50, 40, 10]   # C
]

plot_stacked_histogram(data)
