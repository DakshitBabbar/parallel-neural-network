/*
Fast Artificial Neural Network Library (fann)
Copyright (C) 2003-2016 Steffen Nissen (steffen.fann@gmail.com)

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include "fann.h"

int main()
{
	const unsigned int num_input = 2;
	const unsigned int num_output = 1;
	const unsigned int num_layers = 3;
	const unsigned int num_neurons_hidden = 900;
	const float desired_error = (const float) 0.001;
	const unsigned int max_epochs = 500000;
	const unsigned int epochs_between_reports = 1000;

	struct fann *ann = fann_create_standard(num_layers, num_input, num_neurons_hidden, num_output);

	// int j=0;
	// for(; j<50; j++){
	// 	printf("%f\n", ann->weights[j]);
	// }

	fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

	//printf("lol3\n");
	fann_train_on_file(ann, "xor.data", max_epochs, epochs_between_reports, desired_error);
	//=======================================DEBUG==================================================//
	// struct fann_train_data *data = fann_read_train_from_file("xor.data");
	// printf("aaaa start\n");
	// int i=0 ;
	// for(; i<1 ; i++)
	// {
	// 	fann_run(ann, data->input[i]);
	// 	fann_compute_MSE(ann, data->output[i]);
	// 	// fann_backpropagate_MSE(ann);
	// 	//do not uncomment coz its included in backpropagate//fann_update_slopes_batch(ann, NULL, NULL);
	// }
	// printf("aaaa complete\n");
	// struct fann_layer *last_l
	// struct fann_layer *last_layer, *layer_it;
	// struct fann_neuron *last_neuron, *neuron_it;
	
	// last_layer = ann->last_layer;
	// int l=1;
  	// for (layer_it = ann->first_layer; layer_it != last_layer; layer_it++){
	// 	last_neuron = layer_it->last_neuron;
	// 	int n = 1;
	// 	for (neuron_it = layer_it->first_neuron; neuron_it != last_neuron-1; neuron_it++){
	// 		printf("layer %d, neuron %d, value %f \n",l, n, neuron_it->value);
	// 		n++;
	// 	}
	// 	l++;
	// }
	//=======================================DEBUG==================================================//

	fann_save(ann, "xor_float.net");

	fann_destroy(ann);

	return 0;
}
