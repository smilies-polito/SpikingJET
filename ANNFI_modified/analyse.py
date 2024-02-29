import torch
import numpy as np
import pandas as pd

def normalize_tensor(x):
        # Calculate min and max values
    min_val = torch.min(x)
    max_val = torch.max(x)

    # Normalize tensor between 0 and 1
    x = (x - min_val) / (max_val - min_val)

    return x

def cross_entropy(tensor1, tensor2):

    return -torch.sum(tensor1 * torch.log(tensor2 + 1e-10))



def assess(fault_id, batch_id):
    fault_path = '/home/ldapusers/gogebakan/snn_fault_injection/ANNFI_modified/output/CSNN/pt/faulty/batch_size_8/fault_'+str(fault_id)+'_batch_'+str(batch_id)+'.pt'
    faulty = torch.load(fault_path)

    clean_path = '/home/ldapusers/gogebakan/snn_fault_injection/ANNFI_modified/output/CSNN/pt/clean/batch_size_8/batch_'+str(batch_id)+'.pt'
    clean = torch.load(clean_path)

    label_path = '/home/ldapusers/gogebakan/snn_fault_injection/ANNFI_modified/output/CSNN/pt/label/batch_size_8/batch_'+str(batch_id)+'.pt'
    true_labels = torch.load(label_path)

    true_labels_one_shot = torch.eye(11)[true_labels]





    # print('faulty.size()', faulty.size())
    # print('clean.size()', clean.size())
    # print('label.size()', true_labels.size())
    # print('true_label_one_shot.size()', true_labels_one_shot.size())
    # print('clean[0:2,:]', clean[0:2,:])
    # print('faulty[0:2,:]', faulty[0:2,:])
    # print( clean == faulty)
    # print(true_labels_one_shot)



    # Calculate cross-entropy loss
    cross_entropy_loss = torch.nn.functional.kl_div(faulty, clean)
    true_cross_entropy_loss = torch.nn.functional.kl_div(true_labels_one_shot, clean)

    # Calculate accuracy
    faulty_labels = torch.argmax(faulty, dim=1)
    clean_labels = torch.argmax(clean, dim=1)

    diff_accuracy = torch.mean((faulty_labels == clean_labels).float())
    clean_accuracy = torch.mean((true_labels == clean_labels).float())

    print("diff Cross-entropy Loss:", cross_entropy_loss.item())
    print("true Cross-entropy Loss:", true_cross_entropy_loss.item())
    print("Diff Accuracy:", diff_accuracy.item())
    print("Clean Accuracy:", clean_accuracy.item())
    print('\n')



# assess(18,0)
# assess(18,1)
# assess(18,2)
# assess(18,3)
# assess(36,1)
# assess(44,0)
    

fault_list_path = '/home/ldapusers/gogebakan/snn_fault_injection/ANNFI_modified/output/fault_list/CSNN/51196_fault_list.csv'
fault_list = pd.read_csv(fault_list_path)
print(fault_list)
print(fault_list["Layer"].unique())

conv1_df = fault_list[fault_list['Layer'] == 'conv1.weight']
print(conv1_df['Injection'].tolist())

layer_id = {}
for layer in fault_list["Layer"].unique():
    conv1_df = fault_list[fault_list['Layer'] == layer]
    layer_id[layer] = conv1_df['Injection'].tolist()

print(layer_id.keys())

print(layer_id['conv2.weight'])


