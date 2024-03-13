import os
import numpy as np
import matplotlib.pyplot as plt
import csv
def convert_to_float_nested_x100(data):
    return [[float(x)*100 for x in sublist] for sublist in data]

def read_csv_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row_num, row in enumerate(csv_reader):
            if row_num not in (1, 2):  # Skip second and third rows
                modified_row = row[1:]  # Exclude the first column
                if row_num == 8: #last row
                    data.insert(2, modified_row)
                else:
                    data.append(modified_row)
    # Transpose the data
    transposed_data = list(map(list, zip(*data)))
    # Sort the transposed data based on the values in the first row
    sorted_transposed_data = sorted(transposed_data, key=lambda x: x[0])
    # Transpose the sorted data back to its original form
    sorted_data = list(map(list, zip(*sorted_transposed_data)))
    label = sorted_data[0]
    data = convert_to_float_nested_x100(sorted_data[1:])
   
    data[5] = list(np.array(data[5]) - np.array(data[4]))
    data[4] = list(np.array(data[4]) - np.array(data[3]))
             
    return label, data

def plot_stacked_histogram(label, data):
    params = []
    layers = set()
    for lab in label:
        if lab != '':
            params.append(lab.split('.')[1])
            layers.add(lab.split('.')[1])
        else:
            params.append(lab)    
    categories = ['Fully Masked', 'Masked', 'SDC-1', 'SDC-5%', 'SDC-10%', 'SDC-20%']
    num_categories = len(categories)

    # Transpose the data to have each column representing a category
    data_transposed = np.array(data).T
    bottom = np.zeros(len(data))

    # Grouping the data into three sets for each category
    group_data = [data_transposed[i:i+3] for i in range(0, len(data_transposed), 3)]
    
    plt.figure(figsize=(20, 6))
    k=0
    for i, group in enumerate(group_data):
        
        for j, cat_data in enumerate(group):
            
            plt.bar(range(len(data)), cat_data, bottom=bottom, label=categories[k], alpha=0.7)
            k+=1
            bottom += cat_data
        bottom += 0.05  # Adding space between groups

    plt.ylabel('Percentage')
    plt.title('DVSGesture - Fault Injection Campaign Outcomes')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(range(len(params)), labels=params)
    plt.text(0, -10, 'Conv1')
    plt.text(3, -10, 'Conv2')
    plt.text(6, -10, 'Fully Connected')
    plt.text(10, -10, 'Lif1')
    plt.text(15, -10, 'Lif2')
    plt.text(20, -10, 'Lif3')
    plt.savefig(os.path.join(os.path.dirname(__file__), 'stacked_histogram.png'))
    plt.show()

# Example data: Each column represents a category, and each row represents a column
'''
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
'''
file_path = '3_fault_list_result_batch_size_64.csv'  # Replace 'data.csv' with the path to your CSV file
label, data = read_csv_file(file_path)
data = list(np.array(data).T)

prev = label[0].split('.')[0]
for i, lab in enumerate(label):
    if lab.split('.')[0] != prev:
        prev = lab.split('.')[0]
        label.insert(i, '')
        data.insert(i, [0,0,0,0,0,0])
        
plot_stacked_histogram(label, data)