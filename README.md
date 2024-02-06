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
```

**_NOTE:_** For SNN layers, learn_beta and learn_threshold values must be "True" in order to access it even though it isn't updated during the training.  

3. DVSGesture_model\
&nbsp;&nbsp;&nbsp;&nbsp; We will develop CSNN architectures here. We have already one model that is working well but additional experiments related to model can be done here.

```sh
   python3 class_CSNN_DVSGesture.py
```

## TODO


### Prior
 - [ ] Run Fault injection with CSNN_DVSGesture
    - [x] Create a singularity container that can be used for ANNFI and CSNN_DVSGesture models
    - [x] Understand how to place our module to existing framework
        - [x] Modify codes for data loader
        - [x] Modify codes for model architecture (e.g. include interpolation on forward pass)
    - [x] Check beta and membrane threshold values
        - [x] Right now, there is one beta value for each layer. Modify it so that there is a beta value for each neuron.
    - [x] Inject fault on membrane potentials.
    - [x] Inject fault on beta values.


### Secondary
 - [ ] Modify train scripts as they can be used with argument parser. The aim is to run the same script with different parameters at the same time in different containers or servers.
