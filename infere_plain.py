
# Inference with integers mod t_i. Use of the Chinese Remainder Theorem to encode large numbers.

import numpy as np
from wrapper import t_list

t_size = len(t_list)

# WEIGHTS
dense1_kernel = np.load("./nn_data/dense1_kernel.npy")
dense1_bias = np.load("./nn_data/dense1_bias.npy")
dense2_kernel = np.load("./nn_data/dense2_kernel.npy")
dense2_bias = np.load("./nn_data/dense2_bias.npy")
conv_kernel = np.load("./nn_data/conv_kernel.npy")
conv_bias = np.load("./nn_data/conv_bias.npy")

# INPUT AND OUTPUT DATA
plain_input = np.load("./output_data/plain_layer_0.npy")
examples_count = plain_input.shape[0]
plain_output = np.empty((examples_count,845,5), dtype=np.uint64)



# LAYER 1: convolution + flatten
print("Computing layer 1/5 (with percentage of progress)...")
last_one = -1
for axis0_index in range(examples_count):
	temp = (axis0_index*100)//examples_count
	if (last_one!=temp):
		last_one = temp
		print(str(temp)+"%")
	for filter_index in range(5):
		for x_output_index in range(13):
			for y_output_index in range(13):
				for t_index in range(t_size):
					temp = 0
					for x_filter_index in range(5):
						for y_filter_index in range(5):
							temp = temp + \
								plain_input[axis0_index, x_output_index*2+x_filter_index, y_output_index*2+y_filter_index, t_index].item() * \
								conv_kernel[x_filter_index, y_filter_index, filter_index, t_index].item()
					temp = temp + conv_bias[filter_index, t_index].item()
					temp = temp % t_list[t_index]
					plain_output[axis0_index, filter_index + (y_output_index*5) + (x_output_index*65), t_index] = temp 
plain_input = None
np.save("./output_data/plain_layer_1", plain_output)
print("100%")



# LAYER 2: square activation function
print("Computing layer 2/5...")
plain_output = plain_output*plain_output
for t_index in range(t_size):
	plain_output[...,t_index] = plain_output[...,t_index] % t_list[t_index]
np.save("./output_data/plain_layer_2", plain_output)



# LAYER 3: fully connected layer
print("Computing layer 3/5...")
temp = np.empty((examples_count,100,t_size), dtype=np.uint64)
for t_index in range(t_size):
	temp[...,t_index] = (plain_output[...,t_index].dot(dense1_kernel[...,t_index])) % t_list[t_index]
	temp[...,t_index] = (temp[...,t_index] + dense1_bias[...,t_index]) % t_list[t_index]
plain_output = temp
temp = None
np.save("./output_data/plain_layer_3", plain_output)



# LAYER 4: square activation function
print("Computing layer 4/5...")
plain_output = plain_output*plain_output
for t_index in range(t_size):
	plain_output[...,t_index] = plain_output[...,t_index] % t_list[t_index]
np.save("./output_data/plain_layer_4", plain_output)



# LAYER 5: fully connected layer
temp = np.empty((examples_count,10,t_size), dtype=np.uint64)
for t_index in range(t_size):
	temp[...,t_index] = (plain_output[...,t_index].dot(dense2_kernel[...,t_index])) % t_list[t_index]
	temp[...,t_index] = (temp[...,t_index] + dense2_bias[...,t_index]) % t_list[t_index]
plain_output = temp
temp = None
np.save("./output_data/plain_layer_5", plain_output)



print("Done. Results stored in ./output_data/")