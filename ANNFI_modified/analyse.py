import torch
import numpy as np
import pandas as pd
import os
import argparse

def SDC_percentage(faulty_batch, clean_batch, lower_th, higher_th):
    # Calculates the SDC-percentage score (SDC_10%, SDC_20%)
    clean_labels, clean_index = torch.max(clean_batch, dim=1) 
    faulty_labels, faulty_index = torch.max(faulty_batch, dim=1)

    same_ind = torch.eq(clean_index, faulty_index)
    diff_vec = torch.abs(clean_labels - faulty_labels)/clean_labels
    lower_vec = diff_vec < higher_th
    higher_vec = diff_vec > lower_th

    res_per = same_ind & lower_vec & higher_vec
    score = torch.mean(res_per.float())

    return score.item()

def SDC_interval(faulty_batch, clean_batch):
    # Calculates the SDC-percentage score (SDC_10%, SDC_20%)
    clean_labels, clean_index = torch.max(clean_batch, dim=1) 
    faulty_labels, faulty_index = torch.max(faulty_batch, dim=1)

    same_ind = torch.eq(clean_index, faulty_index)
    same_label = torch.eq(clean_labels, faulty_labels)
    diff_vec = 100*torch.abs(clean_labels - faulty_labels)/clean_labels
    
    less_5 = diff_vec < 5
    less_10 = diff_vec < 10
    less_20 = diff_vec < 20

    more_5 = diff_vec > 5
    more_10 = diff_vec > 10
    more_20 = diff_vec > 20

    fully_masked_bool = same_ind & same_label
    fully_masked_score = torch.mean(fully_masked_bool.float())

    SDC_1_score = 1 - torch.mean(same_ind.float())
    
    SDC_0_5_bool = same_ind & less_5 & torch.logical_not(same_label)
    SDC_0_5_score = torch.mean(SDC_0_5_bool.float())

    SDC_5_10_bool = same_ind & less_10 & more_5
    SDC_5_10_score = torch.mean(SDC_5_10_bool.float())

    SDC_10_20_bool = same_ind & less_20 & more_10
    SDC_10_20_score = torch.mean(SDC_10_20_bool.float())

    SDC_20_bool = same_ind & more_20
    SDC_20_score = torch.mean(SDC_20_bool.float())

    return fully_masked_score.item(), SDC_1_score.item(), SDC_0_5_score.item(), SDC_5_10_score.item(), SDC_10_20_score.item(), SDC_20_score.item()

def SDC(faulty_batch, clean_batch):
    # Calculates the SDC-percentage score (SDC_10%, SDC_20%)
    clean_labels, clean_index = torch.max(clean_batch, dim=1) 
    faulty_labels, faulty_index = torch.max(faulty_batch, dim=1)

    same_ind = torch.eq(clean_index, faulty_index)
    same_label = torch.eq(clean_labels, faulty_labels)
    diff_vec = 100*torch.abs(clean_labels - faulty_labels)/clean_labels
    
    less_5 = diff_vec < 5
    less_10 = diff_vec < 10
    less_20 = diff_vec < 20

    # more_5 = diff_vec > 5
    # more_10 = diff_vec > 10
    more_20 = diff_vec > 20

    fully_masked_bool = same_ind & same_label
    fully_masked_score = torch.mean(fully_masked_bool.float())

    SDC_1_score = 1 - torch.mean(same_ind.float())
    
    SDC_5per_bool = same_ind & less_5 & torch.logical_not(same_label)
    SDC_5per_score = torch.mean(SDC_5per_bool.float())

    SDC_10per_bool = same_ind & less_10 & torch.logical_not(same_label)# & more_5
    SDC_10per_score = torch.mean(SDC_10per_bool.float())

    SDC_20per_bool = same_ind & less_20 & torch.logical_not(same_label)# & more_10
    SDC_20per_score = torch.mean(SDC_20per_bool.float())

    masked_bool = same_ind & more_20
    masked_score = torch.mean(masked_bool.float())

    return fully_masked_score.item(), SDC_1_score.item(), SDC_5per_score.item(), SDC_10per_score.item(), SDC_20per_score.item(), masked_score.item()

def faulty_vs_ground(faulty_batch, labels):

    _, faulty_index = torch.max(faulty_batch, dim=1)
    same_ind = torch.eq(faulty_index,labels)
    acc = torch.mean(same_ind.float())

    return acc.item()



def get_faultID_per_layer(csv_path):
    
    PATH = os.getcwd()
    fault_list_path =PATH + '/output/fault_list' + csv_path
    df_fault = pd.read_csv(fault_list_path)

    ID_dict = {}
    for layer in df_fault["Layer"].unique():
        layer_df = df_fault[df_fault['Layer'] == layer]
        ID_dict[layer] = layer_df['Injection'].tolist()
    
    return ID_dict, df_fault


parser = argparse.ArgumentParser(description='Analyze fault injection results',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--fault-list', '-fl', type=str,
                    help='File name of the fault list')
parser.add_argument('--batch-size', '-b', type=int, default=64,
                    help='Test set batch size')
parser.add_argument('--network-name', '-n', type=str,
                    required=True,
                    help='Target network',
                    choices=['ResNet20', 'ResNet32', 'ResNet44', 'ResNet56', 'ResNet110', 'ResNet1202',
                                'DenseNet121', 'CSNN', 'NMNIST','SHD'])

parsed_args = parser.parse_args()

# batch_size = 16
# model_name = 'CSNN'
# fault_list_name = '51196_fault_list.csv'

batch_size = parsed_args.batch_size
model_name = parsed_args.network_name
fault_list_name = parsed_args.fault_list

ID_dict, df_fault = get_faultID_per_layer('/' + model_name + '/' + fault_list_name) #'/CSNN/51196_fault_list.csv'

PATH = os.getcwd()
faulty_path_prefix = PATH + '/output/' + model_name + '/pt/faulty/batch_size_' + str(batch_size)
clean_path_prefix = PATH + '/output/' + model_name + '/pt/clean/batch_size_' + str(batch_size)
label_path_prefix = PATH + '/output/' + model_name + '/pt/label/batch_size_' + str(batch_size)

sdc_1_dict = {}
fully_masked_dict = {}
score = {}

for layer in ID_dict.keys():
    count = 0
    tot_fully_masked = 0
    tot_SDC_1_score = 0
    tot_SDCper_0_5 = 0
    tot_SDCper_5_10 = 0
    tot_SDCper_10_20 = 0
    tot_SDCper_20 = 0
    tot_facc = 0
    tot_acc = 0

    print(str(len(ID_dict[layer])) + ' fault injected into ' + layer)

    for fault_id in ID_dict[layer]:

        
        faulty_path_list = [filename for filename in os.listdir(faulty_path_prefix) if filename.startswith('fault_' + str(fault_id) + '_batch_')]
        if layer == 'fc1.bias':
            print(faulty_path_list)
        # print('/fault_' + str(fault_id) + '_batch_')

        for batch_path in faulty_path_list:
            batch_id = batch_path.split('_')[-1].split('.')[0]

            clean_path = clean_path_prefix + '/batch_'+str(batch_id)+'.pt'
            faulty_path = faulty_path_prefix + '/' +batch_path
            label_path = label_path_prefix + '/batch_' + str(batch_id)+'.pt'

            clean = torch.load(clean_path)
            faulty = torch.load(faulty_path)
            label = torch.load(label_path)
        
            fully_masked,SDC_1_score,SDCper_0_5,SDCper_5_10,SDCper_10_20,SDCper_20 = SDC_interval(faulty, clean)
            # fully_masked,SDC_1_score,SDCper_0_5,SDCper_5_10,SDCper_10_20,SDCper_10_20 = SDC(faulty, clean)
            facc = faulty_vs_ground(faulty,label)

            acc = faulty_vs_ground(clean,label)


            count +=1
            tot_fully_masked += fully_masked
            tot_SDC_1_score += SDC_1_score
            tot_SDCper_0_5 += SDCper_0_5
            tot_SDCper_5_10 += SDCper_5_10
            tot_SDCper_10_20 += SDCper_10_20
            tot_SDCper_20 += SDCper_20
            tot_facc += facc
            tot_acc += acc

            if count % 10000 == 0:
                check1 = tot_SDC_1_score/count
                check2 = tot_fully_masked/count
                check3 = tot_SDCper_0_5/count
                print('Checkpoint SDC_1_score at {} is {}'.format(count, check1))
                print('Checkpoint tot_fully_masked at {} is {}'.format(count, check2))
                print('Checkpoint tot_SDCper_0_5 at {} is {}'.format(count, check3))
                print()

    print('SDC_1 score for ' + layer + ' is ' + str(tot_SDC_1_score/count))
    print('SDC_5per score for ' + layer + ' is ' + str(tot_SDCper_0_5/count))
    # print('SDC_10per score for ' + layer + ' is ' + str(tot_SDC_10per_score/count))
    # print('SDC_15per score for ' + layer + ' is ' + str(tot_SDC_15per_score/count))
    # print('SDC_20per score for ' + layer + ' is ' + str(tot_SDC_20per_score/count))

    sdc_1_dict[layer] = tot_SDC_1_score/count
    fully_masked_dict[layer] = tot_fully_masked/count
    if count == 0:
        print(layer)
    score[layer] = int(len(ID_dict[layer])),int(count), tot_fully_masked/count, tot_SDC_1_score/count, tot_SDCper_0_5/count, tot_SDCper_5_10/count, tot_SDCper_10_20/count, tot_SDCper_20/count, tot_facc/count, tot_acc/count
    print(sdc_1_dict)
    print(fully_masked_dict)
    print('\n')
    print('\n')

df_score = pd.DataFrame(data=score, index=['# of faults','# of injections','fully_masked','SDC_1', 'SDC_0_5%', 'SDC_5_10%', 'SDC_10_20%', 'SDC_20','facc', 'acc'])
print(df_score)

PATH = os.getcwd()
result_name = fault_list_name.split('.')[0]
result_path =PATH + '/output/fault_list' + '/' + model_name + '/' + result_name + '_result_batch_size_' + str(batch_size) + '.csv'
df_score.to_csv(result_path, index=True)

