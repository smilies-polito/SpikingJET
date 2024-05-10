import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import csv


def convert_to_float_nested_x100(data):
    return [[float(x)*100 for x in sublist] for sublist in data]

def adjust(data):
    matrice_np = np.array(data)
    somme_colonne = np.sum(matrice_np, axis=0)
    return list(matrice_np * (100/somme_colonne))

def read_csv_file(file_path, ds):
    data = []
    facc = []
    custom_order = []
    custom_order_csnn = ["conv1.bias", "conv1.weight", "lif1.beta", "lif1.potential", "lif1.spike", "lif1.threshold", "conv2.bias", "conv2.weight", "lif2.beta", "lif2.potential", "lif2.spike", "lif2.threshold", "fc1.bias", "fc1.weight", "lif3.beta", "lif3.potential", "lif3.spike", "lif3.threshold"]
    custom_order_nmnist = ["fc1.bias", "fc1.weight", "lif1.beta", "lif1.potential", "lif1.spike", "lif1.threshold", "fc2.bias", "fc2.weight", "lif2.beta", "lif2.potential", "lif2.spike", "lif2.threshold"]
    custom_order_shd = ["fc1.bias", "fc1.weight", "fb1.bias", "fb1.weight", "lif1.beta", "lif1.potential", "lif1.spike", "lif1.threshold", "fc2.bias", "fc2.weight", "lif2.beta", "lif2.potential", "lif2.spike", "lif2.threshold"]
    if ds == 1:
        custom_order = custom_order_csnn
    elif ds == 2:
        custom_order = custom_order_nmnist
    else:
        custom_order = custom_order_shd

    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        last = []
        for row_num, row in enumerate(csv_reader):
            if row_num not in (1, 2, 10):  # Skip second and third rows
                modified_row = row[1:]  # Exclude the first column
                if row_num == 4:
                    last = modified_row
                else:
                    data.append(modified_row)
        data.insert(6, last)        
    
    # Transpose the data
    transposed_data = list(map(list, zip(*data)))
    # Sort the transposed data based on the values in the first row
    sorted_transposed_data = sorted(transposed_data, key=lambda x: custom_order.index(x[0]))
    # Transpose the sorted data back to its original form
    sorted_data = list(map(list, zip(*sorted_transposed_data)))

    label = sorted_data[0]
    data = convert_to_float_nested_x100(sorted_data[1:-1])
    data = adjust(data)
    facc = sorted_data[-1]
    facc = [float(x)*100 for x in facc]
   
    #data[5] = list(np.array(data[5]) - np.array(data[4]))
    #data[4] = list(np.array(data[4]) - np.array(data[3]))
             
    return label, data, facc

def plot_stacked_histogram(label, data, ds):
    params = []
    layers = set()
    for lab in label:
        if lab != '':
            params.append(lab.split('.')[1])
            layers.add(lab.split('.')[1])
        else:
            params.append(lab)    
    categories = ['Masked', 'SDC_0-5%', 'SDC_5-10%', 'SDC_10-20%', 'SDC_20%', 'SDC-1']
    num_categories = len(categories)
    colors = ['blue', 'skyblue', 'palegoldenrod', 'orange', 'coral', 'red']

    # Transpose the data to have each column representing a category
    data_transposed = np.array(data).T
    bottom = np.zeros(len(data))

    # Grouping the data into three sets for each category
    group_data = [data_transposed[i:i+3] for i in range(0, len(data_transposed), 3)]
    
    plt.figure(figsize=(20, 4))
    k=0
    for i, group in enumerate(group_data):
        
        for j, cat_data in enumerate(group):
            
            plt.bar(range(len(data)), cat_data, bottom=bottom, label=categories[k], color=colors[k],  alpha=0.7)
            k+=1
            bottom += cat_data
        bottom += 0.05  # Adding space between groups
    plt.axhline(y=20, color='grey', linestyle='--')
    plt.axhline(y=40, color='grey', linestyle='--')
    plt.axhline(y=60, color='grey', linestyle='--')
    plt.axhline(y=80, color='grey', linestyle='--')
    plt.ylabel('Percentage',fontsize=13)
    plt.legend(loc='center left', fontsize=14, bbox_to_anchor=(1, 0.5))
    plt.xticks(range(len(params)), labels=params)
    if ds == 1:
        plt.title('DVSGesture - Fault Injection Campaign Outcomes')
        plt.text(0, -12, 'Conv1', fontsize=13, weight='bold')
        plt.text(4, -12, 'Lif1', fontsize=13, weight='bold')
        plt.text(8, -12, 'Conv2', fontsize=13, weight='bold')
        plt.text(12, -12, 'Lif2', fontsize=13, weight='bold')
        plt.text(16, -12, 'FC1', fontsize=13, weight='bold')
        plt.text(20, -12, 'Lif3', fontsize=13, weight='bold')
    elif ds == 2:
        plt.title('NMNIST - Fault Injection Campaign Outcomes')
        plt.text(0, -13, 'FC1', fontsize=13, weight='bold')
        plt.text(4, -13, 'Lif1', fontsize=13, weight='bold')
        plt.text(8, -13, 'FC2', fontsize=13, weight='bold')
        plt.text(12, -13, 'Lif2', fontsize=13, weight='bold')
    else:
        plt.title('SHD - Fault Injection Campaign Outcomes')
        plt.text(0, -12, 'FC1', fontsize=13, weight='bold')
        plt.text(3, -12, 'FB1', fontsize=13, weight='bold')
        plt.text(7, -12, 'Lif1', fontsize=13, weight='bold')
        plt.text(11, -12, 'FC2', fontsize=13, weight='bold')
        plt.text(15, -12, 'Lif2', fontsize=13, weight='bold')   
    plt.savefig(os.path.join(os.path.dirname(__file__), str(ds)+'stacked_histogram.pdf'))
    plt.show()

def plot_accuracy_histogram(label, acc, ds):
    params = []
    layers = set()
    for lab in label:
        if lab != '':
            params.append(lab.split('.')[1])
            layers.add(lab.split('.')[1])
        else:
            params.append(lab)    
    
    for i, l in enumerate(label):
        if l == '':
            label[i] = str(i)
    plt.figure(figsize=(20, 4))

    plt.bar(label, acc)

    clean_acc = 0
    if ds == 1:
        clean_acc = 72.26
    elif ds == 2:
        clean_acc = 83.58
    else:
        clean_acc = 65.86           
    plt.axhline(y=clean_acc, color='r', linestyle='--')      
    
    plt.ylabel('Accuracy',fontsize=13)
    plt.axhline(y=20, color='grey', linestyle='--')
    plt.axhline(y=40, color='grey', linestyle='--')
    plt.axhline(y=60, color='grey', linestyle='--')
    plt.xticks(range(len(params)), labels=params)
    if ds == 1:
        plt.text(25, clean_acc, 'Clean Run: ' + str(clean_acc), color='red', fontsize=12, ha='center')
        plt.title('DVSGesture - Fault Injection Campaign Outcomes')
        plt.text(0, -9, 'Conv1', fontsize=13, weight='bold')
        plt.text(4, -9, 'Lif1', fontsize=13, weight='bold')
        plt.text(8, -9, 'Conv2', fontsize=13, weight='bold')
        plt.text(12, -9, 'Lif2', fontsize=13, weight='bold')
        plt.text(16, -9, 'FC1', fontsize=13, weight='bold')
        plt.text(20, -9, 'Lif3', fontsize=13, weight='bold')
    elif ds == 2:
        plt.text(16, clean_acc, 'Clean Run: ' + str(clean_acc), color='red', fontsize=12, ha='center')
        plt.title('NMNIST - Fault Injection Campaign Outcomes')
        plt.text(0, -11, 'FC1', fontsize=13, weight='bold')
        plt.text(4, -11, 'Lif1', fontsize=13, weight='bold')
        plt.text(8, -11, 'FC2', fontsize=13, weight='bold')
        plt.text(12, -11, 'Lif2', fontsize=13, weight='bold')
    else:
        plt.text(19.5, clean_acc, 'Clean Run: ' + str(clean_acc), color='red', fontsize=12, ha='center')
        plt.title('SHD - Fault Injection Campaign Outcomes')
        plt.text(0, -9, 'FC1', fontsize=13, weight='bold')
        plt.text(3, -9, 'FB1', fontsize=13, weight='bold')
        plt.text(7, -9, 'Lif1', fontsize=13, weight='bold')
        plt.text(11, -9, 'FC2', fontsize=13, weight='bold')
        plt.text(15, -9, 'Lif2', fontsize=13, weight='bold')     
    plt.savefig(os.path.join(os.path.dirname(__file__), str(ds)+'acc_histogram.pdf'))
    plt.show()

def percent_formatter(x, pos):
    return f'{x:.0f}%'

def plot_net_hist():
    net = ["DVG", "NMNIST", "SHD"]
    metrics = ["Masked",
                "SDC 1",
                "SDC 0-5%",
                "SDC 5-10%",
                "SDC 10-20%",
                "SDC 20%"]
    DVG = [0.939824059452996,
            0.00835794252468265,
            0.0420577627705893,
            0.0028656692677991,
            0.00286471109032931,
            0.00405272574369095]
    NMNIST = [0.987721561221487,
            0.00251444709930268,
            0.00827136891178419,
            0.000349185138770954,
            0.000430775794206003,
            0.000705783041124706]        
    SHD = [0.91835079223029,
            0.00502401290490409,
            0.0650315930751598,
            0.00428994163009712,
            0.00366994402408312,
            0.00362349574737604]

    fig, axs = plt.subplots(3,2, figsize=(8, 10))        
    for i, ax in enumerate(axs.flat):
        ax.bar(net, [DVG[i]*100, NMNIST[i]*100, SHD[i]*100])
        print([DVG[i]*100, NMNIST[i]*100, SHD[i]*100])
        #ax.set_ylabel('Percentage')
        ax.set_title(metrics[i])
        ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))

        # Save the plot as an image file (e.g., PNG)
    plt.tight_layout()
    plt.savefig('bar_plot.pdf') 

def plot_net_hist():
    net = ["DVG", "NMNIST", "SHD"]
    metrics = ["Masked",
                "SDC 1",
                "SDC 0-5%",
                "SDC 5-10%",
                "SDC 10-20%",
                "SDC 20%"]
    DVG = [0.939824059452996,
            0.00835794252468265,
            0.0420577627705893,
            0.0028656692677991,
            0.00286471109032931,
            0.00405272574369095]
    NMNIST = [0.987721561221487,
            0.00251444709930268,
            0.00827136891178419,
            0.000349185138770954,
            0.000430775794206003,
            0.000705783041124706]        
    SHD = [0.91835079223029,
            0.00502401290490409,
            0.0650315930751598,
            0.00428994163009712,
            0.00366994402408312,
            0.00362349574737604]

    fig, axs = plt.subplots(3,2, figsize=(8, 10))        
    for i, ax in enumerate(axs.flat):
        ax.bar(net, [DVG[i]*100, NMNIST[i]*100, SHD[i]*100])
        print([DVG[i]*100, NMNIST[i]*100, SHD[i]*100])
        #ax.set_ylabel('Percentage')
        ax.set_title(metrics[i])
        ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))

        # Save the plot as an image file (e.g., PNG)
    plt.tight_layout()
    plt.savefig('bar_plot.png')   

def plot_net_stacked_hist():
    net = ["DVG", "NMNIST", "SHD"]
    metrics = ["Masked",
               "SDC 0-5%",
               "SDC 5-10%",
               "SDC 10-20%",
               "SDC 20%",
               "SDC 1"]
    DVG = [0.939824059452996,
           0.0420577627705893,
           0.0028656692677991,
           0.00286471109032931,
           0.00405272574369095,
           0.00835794252468265]
    NMNIST = [0.987721561221487,
              0.00827136891178419,
              0.000349185138770954,
              0.000430775794206003,
              0.000705783041124706,
              0.00251444709930268]        
    SHD = [0.91835079223029,
           0.0650315930751598,
           0.00428994163009712,
           0.00366994402408312,
           0.00362349574737604,
           0.00502401290490409]

    fig, ax = plt.subplots(figsize=(6, 6))        

    colors = ['blue', 'skyblue', 'palegoldenrod', 'orange', 'coral', 'red']
    bottom = [0] * len(net)  # Initialize bottom position for stacking
    for i in range(len(metrics)):
        bar = ax.bar(net, [DVG[i]*100, NMNIST[i]*100, SHD[i]*100], bottom=bottom, label=metrics[i], color=colors[i])
        bottom = [bottom[j] + [DVG[i]*100, NMNIST[i]*100, SHD[i]*100][j] for j in range(len(net))]  # Update bottom position for next bar

    ax.set_ylabel('Percentage')
    ax.set_title('Stacked Histogram of Metrics')
    ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig('stacked_bar_plot.pdf')

'''
for ds in range (1,4):
    file_path_csnn = 'CSNN_3_fault_list_result_batch_size_16.csv'
    file_path_nmnist = 'NMNIST_3_fault_list_result_batch_size_128.csv'
    file_path_shd = 'SHD_3_fault_list_result_batch_size_256.csv'
    file_path = ""
    if ds == 1:
        file_path = file_path_csnn
    elif ds == 2:
        file_path = file_path_nmnist
    else:
        file_path = file_path_shd        
        
    label, data, facc = read_csv_file(file_path, ds)
    data = list(np.array(data).T)

    prev = label[0].split('.')[0]
    for i, lab in enumerate(label):
        if lab.split('.')[0] != prev:
            prev = lab.split('.')[0]
            label.insert(i, '')
            data.insert(i, [0,0,0,0,0,0])
            facc.insert(i, 0)
        
    plot_stacked_histogram(label, data, ds)
    plot_accuracy_histogram(label, facc ,ds)
'''    
plot_net_hist()
plot_net_stacked_hist()

