# SpikingJet: SNN fault injection

In order to run scripts on the server, you should create singularity container for the first time:

```sh
singularity build --fakeroot faulty_snn.sif faulty_snn.def
```

After create it once you don't have recreate it everytime. Whenever you want to run scripts, you should start container:


```sh
   singularity shell --nv faulty_snn.sif
```

## Folder Hierarchy
1. FaultGenerators\
&nbsp;&nbsp;&nbsp;&nbsp; Fault injector scripts. It lncludes fault list generator and weight fault injector.

2. models\
&nbsp;&nbsp;&nbsp;&nbsp; This folder contains model script that we want to inject faults and their corresponding pretrained weights. Note that, there should be slight modification after each spiking layer in order to inject faults. 

3. output\
&nbsp;&nbsp;&nbsp;&nbsp; This folder contains output folders. There are one folder for each network whose consist of clean batches, fault injected batches and labels that will be used in the analysis part. In addition to them, there is one folder for fault lists. User doesn@t have to create anything in that folder, it will be managed by the code. If the code is run twice for the same network, injected faults will be overwritten with the new one whereas fault list will be conserved and new one will be added to fault_list folder.


# Fault Injector

The purpose of this fault injector is to inject faults in the network weights (both as stuck-at or bit-flip), run a faulty inference and save the vector score. Injection are performed statistically using formulas from DATE23.

A fault injection can be executed with the following programm:

```sh
   python3 main.py -n CSNN -b 16 --use-cuda
```



**_NOTE:_** For SNN layers, learn_beta and learn_threshold values must be "True" in order to access it even though it isn't updated during the training.  

It is possible to execute inferences with available GPUs sepcifing the argument ```--use-cuda```.

By default, results are saved in ```.pt``` files in the ```output/network_name/pt``` folder. For a faulty run, a single file contains the result of a specific fault in a specific batch. The fault lists can be found in the ```output/fault_list``` folder. Please note that, as of now, it is possible to inject only a fault list at a time: changing the fault list and launching a fault injection campaign for the same network will overwrite previous results.

**Note**: The progress bar shows the percentage of predictions that have chagned as a result of a fault. THIS IS NOT A MEASURE OF ACCURACY LOSS, even if it is related. The behaivour can be changed to check differences in vector score rather than in predicitons.


# Use a different Neural Network model

To use a new custom Neural Network First insert the file with the class of the new NN model inside the folder ```models/```, then instrument the code:
1. Add the FaultInjector parameter in the __init__ function of the model class  ```self.FaultInjector = None```
2. Instrument the forward function, calling the injection function, after each spiking layer, as shown in the code below, call the function ``` self.FaultInjector.injection(mem1, spk1, 'lif1', curr_step) ```. This function takes as parameters, the __membrane potential array__, the __spikes array__, the __name of the layer__ and the __current step__.

   ```
   curr_step = 0
   for step in range(x.shape[0]):
   
      cur1 = self.fc1(x[step])
      spk1, mem1 = self.lif1(cur1, mem1)
      self.FaultInjector.injection(mem1, spk1, 'lif1', curr_step)

      cur2 = self.fc2(spk1)
      spk2, mem2 = self.lif2(cur2, mem2)
      self.FaultInjector.injection(mem2, spk2,'lif2', curr_step)

      spk2_rec.append(spk2)
      curr_step += 1


   ```

# Analyse

After injection fault, results can be analysed by using analyse.py script. The purpose of this analyse is see the SDC scores of the network.


```sh
   python3 analyse.py -n CSNN -b 16 -fl 51196_fault_list.csv
```
**_NOTE:_** The name of the fault list file is generated randomly. It must be checked and run the script accordingly.

Results are saved in the same path with fault list as a csv file. The output format is like the following:

|FIELD1         |fc1.weight            |fc2.weight           |lif1.beta            |lif1.potential       |lif1.threshold      |fc2.bias            |
|---------------|----------------------|---------------------|---------------------|---------------------|--------------------|--------------------|
|# of faults    |15486.0               |240.0                |25.0                 |28.0                 |41.0                |12.0                |
|# of injections|1207908.0             |18720.0              |1950.0               |2184.0               |3198.0              |936.0               |
|fully_masked   |0.9904126338678111    |0.9240660056089743   |0.9072676282051282   |0.8951393658424909   |0.8859492260787992  |0.877128405448718   |
|SDC_1          |0.0015571407859704548 |0.03103382077991453  |0.006935096153846154 |0.019999856913919412 |0.025176868355222013|0.027043269230769232|
|SDC_0_5%       |0.006771314537199853  |0.03959835737179487  |0.07162259615384615  |0.07034111721611722  |0.07413569027517199 |0.08457698985042734 |
|SDC_5_10%      |0.0003157253594644625 |0.0006552150106837606|0.0030488782051282053|0.0027830242673992675|0.003060995153220763|0.0009765625        |
|SDC_10_20%     |0.00035706749810416027|0.001821247329059829 |0.004330929487179488 |0.004356971153846154 |0.004289790494058787|0.003998063568376068|
|SDC_20         |0.0005861179514499449 |0.0028253538995726495|0.006794871794871795 |0.007379664606227106 |0.007387429643527205|0.006276709401709402|
|facc           |0.8350357213670246    |0.8134598691239316   |0.8346995192307692   |0.8275133070054945   |0.8226406152282677  |0.8187182825854701  |
|acc            |0.8358373397435898    |0.8358373397435898   |0.8358373397435898   |0.8358373397435898   |0.8358373397435898  |0.8358373397435898  |


# .pt to .csv

Results file can be converted to csv using the script:
```sh
python pt_to_csv.py -n network-name -b batch-size 
```
Results are saved in the ```output/network_name/csv``` folder. Notice that carrying out operation on the CSV file is going to be more expensive than carrying out the same analysis on .pt files. This format should be used only for data visualization purposes only.


# Reference

@article{gogebakan2024spikingjet,
  title={SpikingJET: Enhancing Fault Injection for Fully and Convolutional Spiking Neural Networks},
  author={Gogebakan, Anil Bayram and Magliano, Enrico and Carpegna, Alessio and Ruospo, Annachiara and Savino, Alessandro and Di Carlo, Stefano},
  journal={arXiv preprint arXiv:2404.00383},
  year={2024}
}

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
