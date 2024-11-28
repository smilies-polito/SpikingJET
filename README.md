# SNN_fault_injection

In order to run scripts on the server, you should create singularity container for the first time:

```sh
singularity build --fakeroot faulty_snn.sif faulty_snn.def
```

After create it once you don't have recreate it everytime. Whenever you want to run scripts, you should start container:


```sh
   singularity shell --nv faulty_snn.sif
```


## Folder Hierarchy

1. ANNFI\
&nbsp;&nbsp;&nbsp;&nbsp; This is the original ANNFI folder. If we lost our way, this is the starting point. Go check this folder and don't change anything here. (Not the exactly original one but can be considered as starting point)

2. ANNFI_modified\
&nbsp;&nbsp;&nbsp;&nbsp; We will develop our algorithms in this folder. Anything can be changed. Check README.md inside of the folder. Additional to these instructions, CSNN model can be inferenced by replacing model name with "CSNN" such as 

```sh
   python3 main.py -n CSNN -b 16
   python3 analyse.py -n CSNN -b 16 -fl 51196_fault_list.csv
```
**_NOTE:_** The name of the fault list file is generated randomly. It must be checked and run the script accordingly.

**_NOTE:_** For SNN layers, learn_beta and learn_threshold values must be "True" in order to access it even though it isn't updated during the training.  

3. DVSGesture_model\
&nbsp;&nbsp;&nbsp;&nbsp; We will develop CSNN architectures here. We have already one model that is working well but additional experiments related to model can be done here.

```sh
   python3 class_CSNN_DVSGesture.py
```

## HOW TO ADD NEW NN:
1) Add in models the file with the model with code instrumentation
    -> Class name NN_DATASET + self.FaultInjector = None + injection in the forward
    -> insert the pretreined weigth file.
    -> if the NN use new types of layer add the new type of layer in the FaultListGenerator line 30
2) utils.py -> modify the parse_args 
            -> modify the load_networks 
3) main.py -> add switch option for the dataloader, check that shuffle parameter is set to false

4) models/utils.py -> add the function fot the dataloader, one for every ds, used for more NN

