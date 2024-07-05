#include <stdbool.h>
#include "fann.h"

//fann_run
//trial 1
fann_type compute_neuron_sum(fann_type* weights, 
                            struct fann_neuron* neurons, 
                            unsigned int num_connections);

//trial 2
void compute_entire_layer(struct fann_neuron* neurons, 
                        unsigned int num_neurons, 
                        struct fann_neuron* neuron_connections, 
                        unsigned int num_neuron_connections, 
                        fann_type* weights,
                        unsigned int num_weights);

//trial 3
void direct_compute_entire_layer(struct fann_neuron* neurons, 
                        unsigned int num_neurons, 
                        struct fann_neuron* neuron_connections, 
                        unsigned int num_neuron_connections, 
                        fann_type* weights,
                        unsigned int num_weights);

//fann_compute_MSE
//trial 1
void compute_MSE_parallel(unsigned int num_neurons,
                        struct fann_neuron *neurons, 
                        fann_type *desired_output, 
                        fann_type *error_it, 
                        float *ann_MSE, 
                        fann_type *ann_bit_fail_limit, 
                        unsigned int *ann_num_bit_fail, 
                        enum fann_errorfunc_enum *ann_train_error_function, 
                        unsigned int *ann_num_MSE);

//fann_backpropagate_MSE
//trial 1
void backpropagate_MSE_parallel(struct fann_neuron *prev_neurons, 
                                fann_type *prev_errors, 
                                fann_type *errors, 
                                fann_type *my_weights,
                                unsigned int num_prev_neurons, 
                                unsigned int num_neurons, 
                                unsigned int num_weights,
                                fann_type *prev_slopes,
                                bool f);

//fann_update_slopes_batch
//trial 1
void update_slopes_parallel(fann_type *prev_slopes, 
                            struct fann_neuron *prev_neurons, 
                            fann_type *my_errors, 
                            unsigned int num_slopes,
                            unsigned int num_prev_neurons,
                            unsigned int num_neurons);   

//fann_train_epoch_irpropm
//trial 1
void fann_train_epoch_parallel(struct fann *ann,  fann_type *input, fann_type *output);                   
