# Requirements 

pip install -r requirements.txt

# Fault Injector

The purpose of this fault injector is to inject faults in the network weights (both as stuck-at or bit-flip), run a faulty inference and save the vector score. Injection are performed statistically using formulas from DATE23.

A fault injection can be executed with the following programm:
```
python main.py -n network-name -b batch-size 
```

Available ANNs are : 'ResNet20', 'ResNet32', 'ResNet44', 'ResNet56', 'ResNet110', 'ResNet1202', 'DenseNet121'

It is possible to execute inferences with available GPUs sepcifing the argument ```--use-cuda```.

By default, results are saved in ```.pt``` files in the ```output/network_name/pt``` folder. For a faulty run, a single file contains the result of a specific fault in a specific batch. The fault lists can be found in the ```output/fault_list``` folder. Please note that, as of now, it is possible to inject only a fault list at a time: changing the fault list and launching a fault injection campaign for the same network will overwrite previous results.

*Note*: The progress bar shows the percentage of predictions that have chagned as a result of a fault. THIS IS NOT A MEASURE OF ACCURACY LOSS, even if it is related. The beavhoiur cna be changed to check differences in vector score rather than in predicitons.

# .pt to .csv

Results file can be converted to csv using the script:
```
python pt_to_csv.py -n network-name -b batch-size 
```
Results are saved in the ```output/network_name/csv``` folder. Notice that carrying out operation on the CSV file is going to be more expensive than carrying out the same analysis on .pt files. This format should be used only for data visualization purposes only.


# Reference

@INPROCEEDINGS{10136998,
  author={Ruospo, A. and Gavarini, G. and de Sio, C. and Guerrero, J. and Sterpone, L. and Reorda, M. Sonza and Sanchez, E. and Mariani, R. and Aribido, J. and Athavale, J.},
  booktitle={2023 Design, Automation & Test in Europe Conference & Exhibition (DATE)}, 
  title={Assessing Convolutional Neural Networks Reliability through Statistical Fault Injections}, 
  year={2023},
  volume={},
  number={},
  pages={1-6},
  doi={10.23919/DATE56975.2023.10136998}}


@INPROCEEDINGS{10173957,
  author={Gavarini, G. and Ruospo, A. and Sanchez, E.},
  booktitle={2023 IEEE European Test Symposium (ETS)}, 
  title={SCI-FI: a Smart, aCcurate and unIntrusive Fault-Injector for Deep Neural Networks}, 
  year={2023},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/ETS56758.2023.10173957}}
