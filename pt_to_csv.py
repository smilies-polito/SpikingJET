import argparse
import csv
import os
import re

import torch
from tqdm import tqdm


def parse_args():

    parser = argparse.ArgumentParser(description='Convert .pt results from csv',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--network-name', '-n', type=str,
                        required=True,
                        help='Target network',
                        choices=['ResNet20', 'ResNet32', 'ResNet44', 'ResNet56', 'ResNet110', 'ResNet1202',
                                 'DenseNet121', 'CSNN'])
    parser.add_argument('--batch-size', '-b', type=int, default=64,
                        help='Test set batch size')

    return parser.parse_args()


def pt_to_csv(network_name: str,
              batch_size: int,
              label: bool,
              faulty: bool,
              input_folder_path: str = None):
    """
    Converts all the .pt files in the folder in a single csv file
    :param network_name: The name of the network
    :param batch_size: The batch size
    :param faulty: Boolean that determines whether the targets are clean or fault output
    :param input_folder_path: Default None. The path of the folder containing the .pt files. If None, the path is
    derived automatically from the other parameters
    """

    # Convert bool to str        
    faulty_clean_str = 'faulty' if faulty else 'clean'
    faulty_clean_str = 'label' if label else faulty_clean_str

    # Build the name of the input folder if not specified
    if input_folder_path is None:
        input_folder_path = f'output/{network_name}/pt/{faulty_clean_str}/batch_size_{batch_size}'


    # Initialize (and create) the output folder path
    output_folder_path = f'output/{network_name}/csv/{faulty_clean_str}/batch_size_{batch_size}'
    os.makedirs(output_folder_path, exist_ok=True)
    output_file_path = f'{output_folder_path}/results.csv'

    number_of_classes = 0

    with open(output_file_path, 'w') as output_file:

        csv_writer = csv.writer(output_file)

        pbar = tqdm(os.listdir(input_folder_path),
                    colour='green',
                    desc=f'Reading {faulty_clean_str} .pt')
       
        for file in pbar: 
            # Append the file only if it is a .pt file
            if file.endswith('.pt'):

                # Open the input file
                tensor = torch.load(f'{input_folder_path}/{file}',
                                    map_location=torch.device('cpu'))   # This avoids loading data to GPU
                if not label:
                    number_of_classes = int(tensor.shape[1])
                        
                    # Append the batch_id information to every tensor
                    batch_id = int(re.findall(r'batch_([0-9]+)', file)[0])
                    batch_ids = torch.ones(tensor.shape[0]).mul(batch_id)
                    tensor = torch.cat([tensor, batch_ids.unsqueeze(dim=1)], dim=1)

                    # Append the image_id
                    image_ids = torch.tensor(range(tensor.shape[0]))
                    tensor = torch.cat([tensor, image_ids.unsqueeze(dim=1)], dim=1)

                    # If faulty, append the fault_id
                    if faulty:
                        fault_id = int(re.findall(r'fault_([0-9]+)', file)[0])
                        fault_ids = torch.ones(tensor.shape[0]).mul(fault_id)
                        tensor = torch.cat([tensor, fault_ids.unsqueeze(dim=1)], dim=1)

                    # Write tensor to csv
                    csv_writer.writerows(tensor.tolist())
                else:
                    csv_writer.writerow(tensor.tolist())

    # Append the header

    # Prepare the header
    if not label:
        header_list = [f'score_class_{i}' for i in range(number_of_classes)] + ['batch_id', 'image_id']

        # Add the fault information for the faulty inferences
        if faulty:
            header_list += ['fault_id']

        csv_header = ', '.join(header_list)
        with open(output_file_path, 'r+') as output_file:
            content = output_file.read()
            output_file.seek(0, 0)
            output_file.write(csv_header.rstrip('\r\n') + '\n' + content)


def main():

    args = parse_args()

    # Convert label
    print("Converting pt to csv for LABELS")
    pt_to_csv(network_name=args.network_name,
              batch_size=args.batch_size,
              faulty=False,
              label=True)
    
    # Convert clean
    print("Converting pt to csv for CLEAN PREDICTIONS")
    pt_to_csv(network_name=args.network_name,
              batch_size=args.batch_size,
              faulty=False,
              label=False)

    # Convert faulty
    print("Converting pt to csv for FAULTY PREDICTIONS")
    pt_to_csv(network_name=args.network_name,
             batch_size=args.batch_size,
             label=False,
             faulty=True)


if __name__ == '__main__':
    main()
