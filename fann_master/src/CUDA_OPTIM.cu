#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "fann.h"
extern "C"{
    #include "CUDA_SUPPORT.h"
}

//fann_run
//tiral 1
__global__ void compute_neuron_sum_kernel(fann_type* weights, struct fann_neuron* neurons, unsigned int num_connections, fann_type* sum){
    int it = threadIdx.x;

    fann_type val = weights[it]*(neurons[it].value);
    atomicAdd(sum, val);
}

extern "C" fann_type compute_neuron_sum(fann_type* weights, struct fann_neuron* neurons, unsigned int num_connections){
    fann_type sum = 0;

    //set memory size
    size_t bytes_weights = num_connections*(sizeof(fann_type));
    size_t bytes_neurons = num_connections*(sizeof(struct fann_neuron));
    size_t bytes_sum = sizeof(fann_type);

    //host pointers are already set

    //host memory is already assigned

    //host data is already intialised

    //set device pointers
    fann_type* d_weights;
    fann_neuron* d_neurons;
    fann_type* d_sum;

    //assign device memory
    cudaMalloc(&d_weights, bytes_weights);
    cudaMalloc(&d_neurons, bytes_neurons);
    cudaMalloc(&d_sum, bytes_sum);

    //copy data from host to device
    cudaMemcpy(d_weights, weights, bytes_weights, cudaMemcpyHostToDevice);
    cudaMemcpy(d_neurons, neurons, bytes_neurons, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum, &sum, bytes_sum, cudaMemcpyHostToDevice);

    compute_neuron_sum_kernel<<<1, num_connections>>>(d_weights, d_neurons, num_connections, d_sum);
    cudaDeviceSynchronize();

    //copy result back to host
    cudaMemcpy(&sum, d_sum, bytes_sum, cudaMemcpyDeviceToHost);

    //free device memory
    cudaFree(d_weights);
    cudaFree(d_neurons);
    cudaFree(d_sum);

    return sum;
}


//trial 2
__global__ void compute_neuron_sum_for_entire_layer_kernel(struct fann_neuron* neurons,  
                                                        struct fann_neuron* neuron_connections, 
                                                        fann_type* weights,
                                                        fann_type* sums){
    int neuron_idx = blockIdx.x;
    int num_connections = blockDim.x;
    int neuron_connection_idx = threadIdx.x;
    int weight_idx = threadIdx.x;

    struct fann_neuron my_neuron = neurons[neuron_idx];

    //do nothing if the neuron is a bias neuron
    if (my_neuron.first_con == my_neuron.last_con) {
        neurons[neuron_idx].value = 1;
        return;
    }

    struct fann_neuron my_neuron_connection = neuron_connections[neuron_connection_idx];
    fann_type my_weight = weights[neuron_idx*num_connections + weight_idx];
    fann_type* my_sum = sums + neuron_idx;

    fann_type val = my_weight*my_neuron_connection.value;

    atomicAdd(my_sum, val);
}

__global__ void compute_entire_layer_kernel(struct fann_neuron* neurons,
                                            fann_type* sums){
    //printf("lol\n");
    int neuron_idx = threadIdx.x;
    
    struct fann_neuron my_neuron = neurons[neuron_idx];

    //do nothing if the neuron is a bias neuron
    if (my_neuron.first_con == my_neuron.last_con) {
        neurons[neuron_idx].value = 1;
        return;
    }

    unsigned int activation_function = my_neuron.activation_function;
    fann_type steepness = my_neuron.activation_steepness;

    fann_type neuron_sum = sums[neuron_idx];

    neuron_sum = fann_mult(steepness, neuron_sum);

    fann_type max_sum = 0;
    max_sum = 150 / steepness;

    if (neuron_sum > max_sum)
    neuron_sum = max_sum;
    else if (neuron_sum < -max_sum)
    neuron_sum = -max_sum;

    neurons[neuron_idx].sum = neuron_sum;

    fann_activation_switch(activation_function, neuron_sum, neurons[neuron_idx].value);
    //printf("neuron value = %f\n",neurons[neuron_idx].value);
}

extern "C" void compute_entire_layer(struct fann_neuron* neurons, 
                                    unsigned int num_neurons, 
                                    struct fann_neuron* neuron_connections, 
                                    unsigned int num_neuron_connections, 
                                    fann_type* weights,
                                    unsigned int num_weights){

    //set memory size
    size_t bytes_neurons = num_neurons*(sizeof(struct fann_neuron));
    size_t bytes_neuron_connections = num_neuron_connections*(sizeof(struct fann_neuron));
    size_t bytes_weights = num_weights*(sizeof(fann_type));
    size_t bytes_sums = num_neurons*(sizeof(fann_type));

    //host pointers are already set

    //host memory is already assigned

    //host data is already intialised

    //set device pointers
    fann_neuron* d_neurons;
    fann_neuron* d_neuron_connections;
    fann_type* d_weights;
    fann_type* d_sums;
    

    //assign device memory
    cudaMalloc(&d_neurons, bytes_neurons);
    cudaMalloc(&d_neuron_connections, bytes_neuron_connections);
    cudaMalloc(&d_weights, bytes_weights);
    cudaMalloc(&d_sums, bytes_sums);

    //copy data from host to device
    cudaMemcpy(d_neurons, neurons, bytes_neurons, cudaMemcpyHostToDevice);
    cudaMemcpy(d_neuron_connections, neuron_connections, bytes_neuron_connections, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, bytes_weights, cudaMemcpyHostToDevice);
    cudaMemset(d_sums, 0, bytes_sums);

    int BLOCKS = num_neurons;
    int THREADS = num_neuron_connections;
    compute_neuron_sum_for_entire_layer_kernel<<<BLOCKS, THREADS>>>(d_neurons, d_neuron_connections, d_weights, d_sums);
    cudaDeviceSynchronize();

    BLOCKS = 1;
    THREADS = num_neurons;
    compute_entire_layer_kernel<<<BLOCKS, THREADS>>>(d_neurons, d_sums);
    cudaDeviceSynchronize();

    //copy result back to host
    cudaMemcpy(neurons, d_neurons, bytes_neurons, cudaMemcpyDeviceToHost);
    // int i=0;
    // for(; i<num_neurons; i++){
    //     printf("neuron value after copy = %f\n", neurons[i].value);
    // }

    //free device memory
    cudaFree(d_neurons);
    cudaFree(d_neuron_connections);
    cudaFree(d_weights);
    //cudaFree(d_sums);
}


//trial 3
__global__ void direct_initialise_neuron_sums(struct fann_neuron* neurons){
    int neuron_idx = threadIdx.x;
    
    neurons[neuron_idx].sum = 0;
    //printf("inside kernel 1\n");
}

__global__ void direct_compute_neuron_sum_for_entire_layer_kernel(struct fann_neuron* neurons,  
                                                                    struct fann_neuron* neuron_connections, 
                                                                    fann_type* weights){
    //printf("inside kernel 2\n");
    int neuron_idx = blockIdx.x;
    int num_connections = blockDim.x;
    int neuron_connection_idx = threadIdx.x;
    int weight_idx = threadIdx.x;

    struct fann_neuron my_neuron = neurons[neuron_idx];

    //do nothing if the neuron is a bias neuron
    if (my_neuron.first_con == my_neuron.last_con) {
        neurons[neuron_idx].value = 1;
        return;
    }

    struct fann_neuron my_neuron_connection = neuron_connections[neuron_connection_idx];
    fann_type my_weight = weights[neuron_idx*num_connections + weight_idx];
    fann_type* my_sum = &neurons[neuron_idx].sum;

    fann_type val = my_weight*my_neuron_connection.value;
    //printf("sum from kernel %f",val);
    atomicAdd(my_sum, val);
}

__global__ void direct_compute_entire_layer_kernel(struct fann_neuron* neurons){
    //printf("lol\n");
    int neuron_idx = threadIdx.x;
    
    struct fann_neuron my_neuron = neurons[neuron_idx];

    //do nothing if the neuron is a bias neuron
    if (my_neuron.first_con == my_neuron.last_con) {
        neurons[neuron_idx].value = 1;
        return;
    }

    unsigned int activation_function = my_neuron.activation_function;
    fann_type steepness = my_neuron.activation_steepness;

    fann_type neuron_sum = neurons[neuron_idx].sum;

    neuron_sum = fann_mult(steepness, neuron_sum);

    fann_type max_sum = 0;
    max_sum = 150 / steepness;

    if (neuron_sum > max_sum)
    neuron_sum = max_sum;
    else if (neuron_sum < -max_sum)
    neuron_sum = -max_sum;

    neurons[neuron_idx].sum = neuron_sum;

    fann_activation_switch(activation_function, neuron_sum, neurons[neuron_idx].value);
    //printf("neuron value = %f\n",neurons[neuron_idx].value);
}

extern "C" void direct_compute_entire_layer(struct fann_neuron* neurons, 
                                    unsigned int num_neurons, 
                                    struct fann_neuron* neuron_connections, 
                                    unsigned int num_neuron_connections, 
                                    fann_type* weights,
                                    unsigned int num_weights){

    //set memory size
    size_t bytes_neurons = num_neurons*(sizeof(struct fann_neuron));
    size_t bytes_neuron_connections = num_neuron_connections*(sizeof(struct fann_neuron));
    size_t bytes_weights = num_weights*(sizeof(fann_type));
    //size_t bytes_sums = num_neurons*(sizeof(fann_type));

    //host pointers are already set

    //host memory is already assigned

    //host data is already intialised

    //set device pointers
    fann_neuron* d_neurons;
    fann_neuron* d_neuron_connections;
    fann_type* d_weights;
    //fann_type* d_sums;
    

    //assign device memory
    cudaMalloc(&d_neurons, bytes_neurons);
    cudaMalloc(&d_neuron_connections, bytes_neuron_connections);
    cudaMalloc(&d_weights, bytes_weights);
    //cudaMalloc(&d_sums, bytes_sums);

    //copy data from host to device
    cudaMemcpy(d_neurons, neurons, bytes_neurons, cudaMemcpyHostToDevice);
    cudaMemcpy(d_neuron_connections, neuron_connections, bytes_neuron_connections, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, bytes_weights, cudaMemcpyHostToDevice);
    //cudaMemset(d_sums, 0, bytes_sums);

    int BLOCKS = 1;
    int THREADS = num_neurons;
    direct_initialise_neuron_sums<<<BLOCKS, THREADS>>>(d_neurons);
    cudaDeviceSynchronize();

    BLOCKS = num_neurons;
    THREADS = num_neuron_connections;
    direct_compute_neuron_sum_for_entire_layer_kernel<<<BLOCKS, THREADS>>>(d_neurons, d_neuron_connections, d_weights);
    cudaDeviceSynchronize();

    BLOCKS = 1;
    THREADS = num_neurons;
    direct_compute_entire_layer_kernel<<<BLOCKS, THREADS>>>(d_neurons);
    cudaDeviceSynchronize();

    //copy result back to host
    cudaMemcpy(neurons, d_neurons, bytes_neurons, cudaMemcpyDeviceToHost);
    // int i=0;
    // for(; i<num_neurons; i++){
    //     printf("neuron value after copy = %f\n", neurons[i].value);
    // }

    //free device memory
    cudaFree(d_neurons);
    cudaFree(d_neuron_connections);
    cudaFree(d_weights);
    //cudaFree(d_sums);
}




//fann_compute_MSE
//trial 1
__global__ void compute_MSE_parallel_kernel(struct fann_neuron *neurons, 
                                            fann_type *desired_output, 
                                            fann_type *error_it, 
                                            float *ann_MSE, 
                                            fann_type ann_bit_fail_limit, 
                                            unsigned int *ann_num_bit_fail, 
                                            enum fann_errorfunc_enum *ann_train_error_function, 
                                            unsigned int *ann_num_MSE){

    int neuron_idx = threadIdx.x;
    fann_neuron my_neuron = neurons[neuron_idx];

    fann_type my_value = my_neuron.value;
    fann_type my_desired_output = desired_output[neuron_idx];

    fann_type diff = my_desired_output - my_value;

    float diff2;

    switch (my_neuron.activation_function) {
        case FANN_LINEAR_PIECE_SYMMETRIC:
        case FANN_THRESHOLD_SYMMETRIC:
        case FANN_SIGMOID_SYMMETRIC:
        case FANN_SIGMOID_SYMMETRIC_STEPWISE:
        case FANN_ELLIOT_SYMMETRIC:
        case FANN_GAUSSIAN_SYMMETRIC:
        case FANN_SIN_SYMMETRIC:
        case FANN_COS_SYMMETRIC:
        {diff /= (fann_type)2.0;
        break;}
        case FANN_THRESHOLD:
        case FANN_LINEAR:
        case FANN_SIGMOID:
        case FANN_SIGMOID_STEPWISE:
        case FANN_GAUSSIAN:
        case FANN_GAUSSIAN_STEPWISE:
        case FANN_ELLIOT:
        case FANN_LINEAR_PIECE:
        case FANN_SIN:
        case FANN_COS:
        break;
    }

    diff2 = (float)(diff * diff);

    atomicAdd(ann_MSE, diff2);
    //printf("ann_MSE = %f\n", *ann_MSE);

    //printf("diff=%f\n", diff);

    /*printf("neuron_diff %f = (%f - %f)[/2], neuron_diff2=%f, sum=%f, MSE_value=%f, num_MSE=%d\n",
    * neuron_diff, *desired_output, neuron_value, neuron_diff2, last_layer_begin->sum,
    * ann->MSE_value, ann->num_MSE); */
    if (fann_abs(diff) >= ann_bit_fail_limit) {
        atomicAdd(ann_num_bit_fail, 1);
    }

    if (*ann_train_error_function) { /* TODO make switch when more functions */
      if (diff < -.9999999)
        diff = -17.0;
      else if (diff > .9999999)
        diff = 17.0;
      else
        diff = (fann_type)log((1.0 + diff) / (1.0 - diff));
    }

    unsigned int activation_function = my_neuron.activation_function;
    fann_type steepness = my_neuron.activation_steepness;
    fann_type value = my_value; 
    fann_type sum = my_neuron.sum;

    fann_type temp = 0;
    switch (activation_function) {
        case FANN_LINEAR:
        case FANN_LINEAR_PIECE:
        case FANN_LINEAR_PIECE_SYMMETRIC:
            {temp = (fann_type)fann_linear_derive(steepness, value);
            break;}
        case FANN_SIGMOID:
        case FANN_SIGMOID_STEPWISE:
            {value = fann_clip(value, 0.01f, 0.99f);
            temp = (fann_type)fann_sigmoid_derive(steepness, value);
            break;}
        case FANN_SIGMOID_SYMMETRIC:
        case FANN_SIGMOID_SYMMETRIC_STEPWISE:
            {value = fann_clip(value, -0.98f, 0.98f);
            temp = (fann_type)fann_sigmoid_symmetric_derive(steepness, value);
            break;}
        case FANN_GAUSSIAN:
            {/* value = fann_clip(value, 0.01f, 0.99f); */
            temp = (fann_type)fann_gaussian_derive(steepness, value, sum);
            break;}
        case FANN_GAUSSIAN_SYMMETRIC:
            {/* value = fann_clip(value, -0.98f, 0.98f); */
            temp = (fann_type)fann_gaussian_symmetric_derive(steepness, value, sum);
            break;}
        case FANN_ELLIOT:
            {value = fann_clip(value, 0.01f, 0.99f);
            temp = (fann_type)fann_elliot_derive(steepness, value, sum);
            break;}
        case FANN_ELLIOT_SYMMETRIC:
            {value = fann_clip(value, -0.98f, 0.98f);
            temp = (fann_type)fann_elliot_symmetric_derive(steepness, value, sum);
            break;}
        case FANN_SIN_SYMMETRIC:
            {temp = (fann_type)fann_sin_symmetric_derive(steepness, sum);
            break;}
        case FANN_COS_SYMMETRIC:
            {temp = (fann_type)fann_cos_symmetric_derive(steepness, sum);
            break;}
        case FANN_SIN:
            {temp = (fann_type)fann_sin_derive(steepness, sum);
            break;}
        case FANN_COS:
            {temp = (fann_type)fann_cos_derive(steepness, sum);
            break;}
        // case FANN_THRESHOLD:
        //     fann_error(NULL, FANN_E_CANT_TRAIN_ACTIVATION);
    }

    error_it[neuron_idx] = temp*diff;
    //printf("temp=%f\n", temp);

    atomicAdd(ann_num_MSE, 1);
    //printf("ann_num_MSE from kernel = %u\n", *ann_num_MSE);
}


extern "C" void compute_MSE_parallel(unsigned int num_neurons,
                                    struct fann_neuron *neurons, 
                                    fann_type *desired_output, 
                                    fann_type *error_it, 
                                    float *ann_MSE, 
                                    fann_type *ann_bit_fail_limit, 
                                    unsigned int *ann_num_bit_fail, 
                                    enum fann_errorfunc_enum *ann_train_error_function, 
                                    unsigned int *ann_num_MSE){
    
    //printf("INITIAL num_MSE lmao= %u\n", *ann_num_MSE);
    //set memory size
    size_t bytes_neurons = num_neurons*(sizeof(struct fann_neuron));
    size_t bytes_desired_output = num_neurons*(sizeof(fann_type));
    size_t bytes_error_it = num_neurons*(sizeof(fann_type));
    size_t bytes_ann_MSE = sizeof(float);
    //size_t bytes_ann_bit_fail_limit = sizeof(fann_type); 
    size_t bytes_ann_num_bit_fail = sizeof(unsigned int);
    size_t bytes_ann_train_error_function = sizeof(enum fann_errorfunc_enum); 
    size_t bytes_ann_num_MSE = sizeof(unsigned int);
    //host pointers are already set

    //host memory is already assigned

    //host data is already intialised

    //set device pointers
    struct fann_neuron *d_neurons;
    fann_type *d_desired_output;
    fann_type *d_error_it;
    float *d_ann_MSE;
    //fann_type *d_ann_bit_fail_limit; 
    unsigned int *d_ann_num_bit_fail;
    enum fann_errorfunc_enum *d_ann_train_error_function;
    unsigned int *d_ann_num_MSE;

    //assign device memory
    cudaMalloc(&d_neurons, bytes_neurons);
    cudaMalloc(&d_desired_output, bytes_desired_output);
    cudaMalloc(&d_error_it, bytes_error_it);
    cudaMalloc(&d_ann_MSE, bytes_ann_MSE);
    //cudaMalloc(&d_ann_bit_fail_limit, bytes_ann_bit_fail_limit); 
    cudaMalloc(&d_ann_num_bit_fail, bytes_ann_num_bit_fail);
    cudaMalloc(&d_ann_train_error_function, bytes_ann_train_error_function);
    cudaMalloc(&d_ann_num_MSE, bytes_ann_num_MSE);

    //copy data from host to device
    cudaMemcpy(d_neurons, neurons,  bytes_neurons, cudaMemcpyHostToDevice);
    cudaMemcpy(d_desired_output, desired_output, bytes_desired_output, cudaMemcpyHostToDevice);
    //printf("seg faulty\n");
    cudaMemcpy(d_error_it, error_it, bytes_error_it, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ann_MSE, ann_MSE, bytes_ann_MSE, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_ann_bit_fail_limit, ann_bit_fail_limit, bytes_ann_bit_fail_limit, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_ann_num_bit_fail, ann_num_bit_fail, bytes_ann_num_bit_fail, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ann_train_error_function, ann_train_error_function, bytes_ann_train_error_function, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ann_num_MSE, ann_num_MSE, bytes_ann_num_MSE, cudaMemcpyHostToDevice);

    int BLOCKS = 1;
    int THREADS = num_neurons;
    compute_MSE_parallel_kernel<<<BLOCKS, THREADS>>>(d_neurons, d_desired_output, d_error_it, d_ann_MSE, *ann_bit_fail_limit, d_ann_num_bit_fail, d_ann_train_error_function, d_ann_num_MSE);
    cudaDeviceSynchronize();

    //copy result back to host
    cudaMemcpy(error_it, d_error_it, bytes_error_it, cudaMemcpyDeviceToHost);
    cudaMemcpy(ann_MSE, d_ann_MSE, bytes_ann_MSE, cudaMemcpyDeviceToHost);
    cudaMemcpy(ann_num_bit_fail, d_ann_num_bit_fail, bytes_ann_num_bit_fail, cudaMemcpyDeviceToHost);
    cudaMemcpy(ann_num_MSE, d_ann_num_MSE, bytes_ann_num_MSE, cudaMemcpyDeviceToHost);
    //printf("FINAL num_MSE lmao= %u\n", *ann_num_MSE);

    //free device memory
    cudaFree(d_neurons);
    cudaFree(d_desired_output);
    cudaFree(d_error_it);
    cudaFree(d_ann_MSE);
    //cudaFree(d_ann_bit_fail_limit); 
    cudaFree(d_ann_num_bit_fail);
    cudaFree(d_ann_train_error_function);
    cudaFree(d_ann_num_MSE);
}

//fann_backpropagate_MSE
//trial 1
__global__ void backpropagate_MSE_parallel_kernel(struct fann_neuron *prev_neurons,
                                                    fann_type *prev_errors, 
                                                    fann_type *errors, 
                                                    fann_type *weights,
                                                    fann_type *prev_slopes,
                                                    bool f){

    int prev_error_idx = blockIdx.x;
    int error_idx = threadIdx.x;
    int weight_idx = error_idx*gridDim.x + prev_error_idx;

    int slope_idx = ((threadIdx.x)*(gridDim.x)) + blockIdx.x;
    int neuron_idx = blockIdx.x;

    fann_type val1 = (errors[error_idx])*(prev_neurons[neuron_idx].value);
    prev_slopes[slope_idx] += val1;
    //printf("slope from kernel = %f from thread %d and block %d\n", val1, threadIdx.x, blockIdx.x);

    if(f){
        return;
    }

    fann_type val = errors[error_idx] * weights[weight_idx];
    atomicAdd(prev_errors + prev_error_idx, val);
}

__global__ void activate_errors(struct fann_neuron *prev_neurons, 
                                fann_type *prev_errors){

    int neuron_idx = threadIdx.x;                                
    fann_neuron my_neuron = prev_neurons[neuron_idx];

    unsigned int activation_function = my_neuron.activation_function;
    fann_type steepness = my_neuron.activation_steepness;
    fann_type value = my_neuron.value; 
    fann_type sum = my_neuron.sum;

    fann_type temp = 0;
    switch (activation_function) {
        case FANN_LINEAR:
        case FANN_LINEAR_PIECE:
        case FANN_LINEAR_PIECE_SYMMETRIC:
            {temp = (fann_type)fann_linear_derive(steepness, value);
            break;}
        case FANN_SIGMOID:
        case FANN_SIGMOID_STEPWISE:
            {value = fann_clip(value, 0.01f, 0.99f);
            temp = (fann_type)fann_sigmoid_derive(steepness, value);
            break;}
        case FANN_SIGMOID_SYMMETRIC:
        case FANN_SIGMOID_SYMMETRIC_STEPWISE:
            {value = fann_clip(value, -0.98f, 0.98f);
            temp = (fann_type)fann_sigmoid_symmetric_derive(steepness, value);
            break;}
        case FANN_GAUSSIAN:
            {/* value = fann_clip(value, 0.01f, 0.99f); */
            temp = (fann_type)fann_gaussian_derive(steepness, value, sum);
            break;}
        case FANN_GAUSSIAN_SYMMETRIC:
            {/* value = fann_clip(value, -0.98f, 0.98f); */
            temp = (fann_type)fann_gaussian_symmetric_derive(steepness, value, sum);
            break;}
        case FANN_ELLIOT:
            {value = fann_clip(value, 0.01f, 0.99f);
            temp = (fann_type)fann_elliot_derive(steepness, value, sum);
            break;}
        case FANN_ELLIOT_SYMMETRIC:
            {value = fann_clip(value, -0.98f, 0.98f);
            temp = (fann_type)fann_elliot_symmetric_derive(steepness, value, sum);
            break;}
        case FANN_SIN_SYMMETRIC:
            {temp = (fann_type)fann_sin_symmetric_derive(steepness, sum);
            break;}
        case FANN_COS_SYMMETRIC:
            {temp = (fann_type)fann_cos_symmetric_derive(steepness, sum);
            break;}
        case FANN_SIN:
            {temp = (fann_type)fann_sin_derive(steepness, sum);
            break;}
        case FANN_COS:
            {temp = (fann_type)fann_cos_derive(steepness, sum);
            break;}
        // case FANN_THRESHOLD:
        //     fann_error(NULL, FANN_E_CANT_TRAIN_ACTIVATION);
    }

    prev_errors[neuron_idx] *= temp;
}

extern "C" void backpropagate_MSE_parallel(struct fann_neuron *prev_neurons, 
                                            fann_type *prev_errors, 
                                            fann_type *errors, 
                                            fann_type *my_weights,
                                            unsigned int num_prev_neurons, 
                                            unsigned int num_neurons, 
                                            unsigned int num_weights,
                                            fann_type *prev_slopes,
                                            bool f){
    
    //set memory size
    size_t bytes_prev_neurons = num_prev_neurons*(sizeof(struct fann_neuron));
    size_t bytes_prev_errors = num_prev_neurons*(sizeof(fann_type));
    size_t bytes_errors = num_neurons*(sizeof(fann_type));
    size_t bytes_weights = num_weights*(sizeof(fann_type));
    size_t bytes_prev_slopes = num_weights*(sizeof(fann_type));

    //host pointers are already set

    //host memory is already assigned

    //host data is already intialised

    //set device pointers
    struct fann_neuron *d_prev_neurons;
    fann_type *d_prev_errors;
    fann_type *d_errors;
    fann_type *d_weights;
    fann_type *d_prev_slopes;
    
    //assign device memory
    cudaMalloc(&d_prev_neurons, bytes_prev_neurons);
    cudaMalloc(&d_prev_errors, bytes_prev_errors);
    cudaMalloc(&d_errors, bytes_errors);
    cudaMalloc(&d_weights, bytes_weights); 
    cudaMalloc(&d_prev_slopes, bytes_prev_slopes);

    //copy data from host to device
    cudaMemcpy(d_prev_neurons, prev_neurons,  bytes_prev_neurons, cudaMemcpyHostToDevice);
    cudaMemcpy(d_prev_errors, prev_errors, bytes_prev_errors, cudaMemcpyHostToDevice);
    cudaMemcpy(d_errors, errors, bytes_errors, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, my_weights, bytes_weights, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_prev_slopes, prev_slopes,  bytes_prev_slopes, cudaMemcpyHostToDevice);

    int BLOCKS = num_prev_neurons;
    int THREADS = num_neurons-1;
    backpropagate_MSE_parallel_kernel<<<BLOCKS, THREADS>>>(d_prev_neurons, d_prev_errors, d_errors, d_weights, d_prev_slopes, f);
    cudaDeviceSynchronize();

    BLOCKS = 1;
    THREADS = num_prev_neurons;
    activate_errors<<<BLOCKS, THREADS>>>(d_prev_neurons, d_prev_errors);
    cudaDeviceSynchronize();

    //copy result back to host
    cudaMemcpy(prev_errors, d_prev_errors, bytes_prev_errors, cudaMemcpyDeviceToHost);
    cudaMemcpy(prev_slopes, d_prev_slopes, bytes_prev_slopes, cudaMemcpyDeviceToHost);

    //free device memory
    cudaFree(d_prev_neurons);
    cudaFree(d_prev_errors);
    cudaFree(d_errors);
    cudaFree(d_weights);
    cudaFree(d_prev_slopes);
}


//fann_update_slopes_batch
//trial 1
__global__ void update_slopes_parallel_kernel(fann_type *prev_slopes, 
                                    fann_neuron *prev_neurons, 
                                    fann_type *errors){
    
    int slope_idx = ((threadIdx.x)*(gridDim.x)) + blockIdx.x;
    int neuron_idx = blockIdx.x;
    int error_idx = threadIdx.x;

    fann_type val = (errors[error_idx])*(prev_neurons[neuron_idx].value);
    prev_slopes[slope_idx] += val;
    //printf("slope from kernel = %f from thread %d and block %d\n", val, threadIdx.x, blockIdx.x);
}

extern "C" void update_slopes_parallel(fann_type *prev_slopes, 
                                    fann_neuron *prev_neurons, 
                                    fann_type *errors, 
                                    unsigned int num_slopes,
                                    unsigned int num_prev_neurons,
                                    unsigned int num_neurons){
    
    //set memory size
    size_t bytes_prev_slopes = num_slopes*(sizeof(fann_type));
    size_t bytes_prev_neurons = num_prev_neurons*(sizeof(struct fann_neuron));
    size_t bytes_errors = num_neurons*(sizeof(fann_type));

    //host pointers are already set

    //host memory is already assigned

    //host data is already intialised

    //set device pointers
    fann_type *d_prev_slopes;
    struct fann_neuron *d_prev_neurons;
    fann_type *d_errors;
    
    //assign device memory
    cudaMalloc(&d_prev_slopes, bytes_prev_slopes);
    cudaMalloc(&d_prev_neurons, bytes_prev_neurons);
    cudaMalloc(&d_errors, bytes_errors);

    //copy data from host to device
    cudaMemcpy(d_prev_slopes, prev_slopes,  bytes_prev_slopes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_prev_neurons, prev_neurons, bytes_prev_neurons, cudaMemcpyHostToDevice);
    cudaMemcpy(d_errors, errors, bytes_errors, cudaMemcpyHostToDevice);

    dim3 BLOCKS = num_prev_neurons;
    int THREADS = num_neurons-1;
    update_slopes_parallel_kernel<<<BLOCKS, THREADS>>>(d_prev_slopes, d_prev_neurons, d_errors);
    cudaDeviceSynchronize();
    //printf("next layer-->\n");

    //copy result back to host
    cudaMemcpy(prev_slopes, d_prev_slopes, bytes_prev_slopes, cudaMemcpyDeviceToHost);
    // int i=0;
    // for(; i<num_slopes; i++){
    //     printf("slopes from outside the kernel, %f\n", prev_slopes[i]);
    // }

    //free device memory
    cudaFree(d_prev_slopes);
    cudaFree(d_prev_neurons);
    cudaFree(d_errors);
}


//fann_train_epoch_irpropm
void fann_train_epoch_parallel(struct fann *ann,  fann_type *input, fann_type *output){
    fann_run(ann, input);
    fann_compute_MSE(ann, output);
    fann_backpropagate_MSE(ann);
    fann_update_slopes_batch(ann, NULL, NULL);
}