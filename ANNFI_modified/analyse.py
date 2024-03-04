import torch
import numpy as np
import pandas as pd
import os
import argparse


def SDC_1(faulty_batch, clean_batch):
    # Calculates the SDC-1 score
    faulty_labels = torch.argmax(faulty_batch, dim=1)
    clean_labels = torch.argmax(clean_batch, dim=1)

    sdc1_score = torch.mean((faulty_labels == clean_labels).float())
    return sdc1_score.item()


def SDC_percentage(faulty_batch, clean_batch, threshold):
    # Calculates the SDC-percentage score (SDC_10%, SDC_20%)
    clean_labels, clean_index = torch.max(clean_batch, dim=1) 
    
    clean_ind_mask = torch.zeros(clean.shape[0],clean.shape[1])
    clean_ind_mask.scatter_(1, clean_index.unsqueeze(1), 1)
    faulty_masked = clean_ind_mask * faulty_batch

    faulty_labels, __ = torch.max(faulty_masked, dim=1)

    difference_vector = 100*(faulty_labels - clean_labels)/clean_labels
    score = torch.mean((abs(difference_vector) < threshold).float())

    return score.item()

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
                                'DenseNet121', 'CSNN'])

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

sdc_1_dict = {}
sdc_5per_dict = {}
score = {}

for layer in ID_dict.keys():
    count = 0
    tot_SDC_1_score = 0
    tot_SDC_5per_score = 0
    tot_SDC_10per_score = 0
    tot_SDC_15per_score = 0
    tot_SDC_20per_score = 0

    print(str(len(ID_dict[layer])) + ' fault injected into ' + layer)

    for fault_id in ID_dict[layer]:

        
        faulty_path_list = [filename for filename in os.listdir(faulty_path_prefix) if filename.startswith('fault_' + str(fault_id) + '_batch_')]
        # print(faulty_path_list)
        # print('/fault_' + str(fault_id) + '_batch_')

        for batch_path in faulty_path_list:
            batch_id = batch_path.split('_')[-1].split('.')[0]

            clean_path = clean_path_prefix + '/batch_'+str(batch_id)+'.pt'
            faulty_path = faulty_path_prefix + '/' +batch_path

            clean = torch.load(clean_path)
            faulty = torch.load(faulty_path)
            
            SDC_1_score = SDC_1(faulty, clean)
            SDC_5per_score = SDC_percentage(faulty, clean, 5)
            SDC_10per_score = SDC_percentage(faulty, clean, 10)
            SDC_15per_score = SDC_percentage(faulty, clean, 15)
            SDC_20per_score = SDC_percentage(faulty, clean, 20)

            count +=1
            tot_SDC_1_score += SDC_1_score
            tot_SDC_5per_score += SDC_5per_score
            tot_SDC_10per_score += SDC_10per_score
            tot_SDC_15per_score += SDC_15per_score
            tot_SDC_20per_score += SDC_20per_score

            # if count % 10000 == 0:
            #     check1 = tot_SDC_1_score/count
            #     check2 = tot_SDC_5per_score/count
            #     print('Checkpoint SDC_1_score at {} is {}'.format(count, check1))
            #     print('Checkpoint SDC_5per_score at {} is {}'.format(count, check2))
            #     print('/n')

    print('SDC_1 score for ' + layer + ' is ' + str(tot_SDC_1_score/count))
    print('SDC_5per score for ' + layer + ' is ' + str(tot_SDC_5per_score/count))
    print('SDC_10per score for ' + layer + ' is ' + str(tot_SDC_10per_score/count))
    print('SDC_15per score for ' + layer + ' is ' + str(tot_SDC_15per_score/count))
    print('SDC_20per score for ' + layer + ' is ' + str(tot_SDC_20per_score/count))

    sdc_1_dict[layer] = tot_SDC_1_score/count
    sdc_5per_dict[layer] = tot_SDC_5per_score/count
    score[layer] = int(len(ID_dict[layer])),int(count), tot_SDC_1_score/count, tot_SDC_5per_score/count, tot_SDC_10per_score/count, tot_SDC_15per_score/count, tot_SDC_20per_score/count
    print(sdc_1_dict)
    print(sdc_5per_dict)
    print('\n')
    print('\n')

df_score = pd.DataFrame(data=score, index=['# of faults','# of injections','SDC_1', 'SDC_5%', 'SDC_10%', 'SDC_15%', 'SDC_20%'])
print(df_score)

PATH = os.getcwd()
result_name = fault_list_name.split('.')[0]
result_path =PATH + '/output/fault_list' + '/' + model_name + '/' + result_name + '_result_batch_size_' + str(batch_size) + '.csv'
df_score.to_csv(result_path, index=True)

