#CUDA Implementation of ANN
#COL730 Assignment 1

**Using the Parallel Code**
The fann_master folder contains the following along with other neccessary contents:
    1. src folder: contains the src code of the entire fann_library modified by us to parallelise the implementation
    2. makefile: used to compile and run the ann developed for mushroom, robot and gene datasets
    3. examples: contains the main code for the training ann

To use the parallel code first run the command
    $module load compiler/cuda/9.2/compilervars
    $make
This will compile all the neccessary code
To train the model for mushroom dataset use the command
    $make run_mushroom
To train the model for robot dataset use the command
    $make run_robot
To train the model for gene dataset use the command
    $make run_gene

**Structure src code**
The functions that are parallelised are present in the following files,
    fann.c
    fann_train.c
These files include the following header file,
    CUDA_SUPPORT.h
Which holds the names of the CUDA functions that are called by the parallelised functions. 
The definition of these CUDA functions along with the kernel calls are given in the following file
    CUDA_OPTIM.cu
Which includes the above mentioned header file in extern "C".

This structure allows us to modify the functions defined in the fann library and call the kernels 
defined in the CUDA files directly from the C files.

Comments in the src code specify the trial number that is being done.
By default, the most optimum version of the parallel implementation will be executed, 
all the other versions are commented out.

For more details refer to the report attached along