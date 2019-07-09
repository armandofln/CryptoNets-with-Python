
# Infere whithin the encrypted polynomials space.

import numpy as np
from wrapper import SEAL

SEALobj = SEAL()
q_list = SEALobj.q_list
k_list = SEALobj.k_list
n_parm = SEALobj.n_parm
enc_poly_size = SEALobj.enc_poly_size
q_size = len(q_list)
t_size = len(SEALobj.t_list)

def to_dtype_object(tensoreuint):
	shape = tensoreuint.shape
	tensoreuint.shape = (tensoreuint.size,)
	new = np.empty((tensoreuint.size,), dtype=object)
	for i in range(tensoreuint.size):
		new[i] = int(tensoreuint[i].item())
	new.shape = shape
	return new

# WEIGHTS
dense1_kernel = np.load("./nn_data/dense1_kernel.npy")
dense1_bias = np.load("./nn_data/dense1_bias.npy")
dense2_kernel = np.load("./nn_data/dense2_kernel.npy")
dense2_bias = np.load("./nn_data/dense2_bias.npy")
conv_kernel = np.load("./nn_data/conv_kernel.npy")
conv_bias = np.load("./nn_data/conv_bias.npy")

# INPUT AND OUTPUT DATA
print("Encrypting the input...")
encrypted_input = np.load("./output_data/plain_layer_0.npy") # not yet encrypted
examples_count = encrypted_input.shape[0]
encrypted_input = SEALobj.encrypt_tensor(encrypted_input) # now it is
np.save("./output_data/enc_layer_0", encrypted_input)
poly_groups_count = encrypted_input.shape[0]//enc_poly_size
encrypted_output = np.empty((encrypted_input.shape[0],845,t_size), dtype=np.uint64)



# LAYER 1: convolution + flatten
print("Computing layer 1/5 (with percentage of progress)...")
for poly_group_index in range(poly_groups_count):
	for s_index in range(2):
		for q_index in range(q_size):
			temp = q_index + (s_index*q_size) + (poly_group_index*2*q_size)
			temp = temp / (poly_groups_count*2*q_size)
			temp = round(temp*1000)
			print(str(temp/10)+"%")
			for n_index in range(n_parm+1):
				axis0_index = n_index + (q_index*(n_parm+1)) + (s_index*q_size*(n_parm+1)) + (poly_group_index*enc_poly_size)
				for filter_index in range(5):
					for x_output_index in range(13):
						for y_output_index in range(13):
							for t_index in range(t_size):
								temp = 0
								for x_filter_index in range(5):
									for y_filter_index in range(5):
										temp = temp + \
											encrypted_input[axis0_index, x_output_index*2+x_filter_index, y_output_index*2+y_filter_index, t_index].item() * \
											conv_kernel[x_filter_index, y_filter_index, filter_index, t_index].item()
								if (n_index==0):
									if (s_index==0):
										temp = temp + ((conv_bias[filter_index, t_index].item())*k_list[t_index][q_index])
								temp = temp % q_list[q_index]
								encrypted_output[axis0_index, filter_index + (y_output_index*5) + (x_output_index*65), t_index] = temp
encrypted_input = None # all data is stored in encrypted_output now
np.save("./output_data/enc_layer_1", encrypted_output)
print("100%")



# LAYER 2: square activation function
print("Computing layer 2/5...")
encrypted_output = SEALobj.square_tensor(encrypted_output)
np.save("./output_data/enc_layer_2", encrypted_output)



# LAYER 3: fully connected layer
print("Computing layer 3/5 (with percentage of progress)...")
dense1_kernel = to_dtype_object(dense1_kernel)
encrypted_output = to_dtype_object(encrypted_output)
## kernel
print("Phase 1/2:")
temp = np.empty((encrypted_output.shape[0],100,t_size), dtype=object)
for t_index in range(t_size):
	print(str((t_index*100)//t_size) + "%")
	temp[...,t_index] = encrypted_output[...,t_index].dot(dense1_kernel[...,t_index])
encrypted_output = temp
temp = None
## % q
print("100%\nPhase 2/2:")
previous_percentage = -1
for axis1 in range(100):
	temp = (axis1*100)//100
	if (previous_percentage!=temp):
		previous_percentage = temp
		print(str(temp)+"%")
	for axis2 in range(encrypted_output.shape[2]):
		for poly_group_index in range(poly_groups_count):
			for size_index in range(2):
				for q_index in range(q_size):
					for n_index in range(n_parm+1):
						axis0 = poly_group_index*enc_poly_size + size_index*q_size*(n_parm+1) + q_index*(n_parm+1) + n_index
						temp = encrypted_output[axis0,axis1,axis2]
						temp = temp % q_list[q_index]
						encrypted_output[axis0,axis1,axis2] = temp
## bias
for axis1 in range(encrypted_output.shape[1]):
	for axis2 in range(encrypted_output.shape[2]):
		for poly_group_index in range(poly_groups_count):
			for q_index in range(q_size):
				axis0 = poly_group_index*enc_poly_size + ((n_parm+1)*q_index)
				temp = encrypted_output[axis0,axis1,axis2]
				temp = temp + dense1_bias[axis1,axis2].item()*k_list[axis2][q_index]
				temp = temp % q_list[q_index]
				encrypted_output[axis0,axis1,axis2] = temp
np.save("./output_data/enc_layer_3", encrypted_output) # !!!! encrypted_output has dtype=object. To load it use the function at the end of the file !!!!
print("100%")



# LAYER 4: square activation function
print("Computing layer 4/5...")
encrypted_output = SEALobj.square_tensor(encrypted_output)
np.save("./output_data/enc_layer_4", encrypted_output)



# LAYER 5: fully connected layer
print("Computing layer 5/5...")
encrypted_output = to_dtype_object(encrypted_output)
dense2_kernel = to_dtype_object(dense2_kernel)
## kernel
temp = np.empty((encrypted_output.shape[0],10,t_size), dtype=object)
for t_index in range(t_size):
	temp[...,t_index] = encrypted_output[...,t_index].dot(dense2_kernel[...,t_index])
encrypted_output = temp
temp = None
## % q
for axis1 in range(encrypted_output.shape[1]):
	for axis2 in range(encrypted_output.shape[2]):
		for poly_group_index in range(poly_groups_count):
			for size_index in range(2):
				for q_index in range(q_size):
					for n_index in range(n_parm+1):
						axis0 = poly_group_index*enc_poly_size + size_index*q_size*(n_parm+1) + q_index*(n_parm+1) + n_index
						temp = encrypted_output[axis0,axis1,axis2]
						temp = temp % q_list[q_index]
						encrypted_output[axis0,axis1,axis2] = temp
## bias
for axis1 in range(encrypted_output.shape[1]):
	for axis2 in range(encrypted_output.shape[2]):
		for poly_group_index in range(poly_groups_count):
			for q_index in range(q_size):
				axis0 = poly_group_index*enc_poly_size + ((n_parm+1)*q_index)
				temp = encrypted_output[axis0,axis1,axis2]
				temp = temp + dense2_bias[axis1,axis2].item()*k_list[axis2][q_index]
				temp = temp % q_list[q_index]
				encrypted_output[axis0,axis1,axis2] = temp
np.save("./output_data/enc_layer_5", encrypted_output) # !!!! encrypted_output has dtype=object. To load it use the function at the end of the file !!!!



# DECRYPT
print("Decrypting the output...")
decrypted_output = SEALobj.decrypt_tensor(encrypted_output, examples_count)
encrypted_output = None
np.save("./output_data/decrypted_layer_5", decrypted_output)



print("Done. Results stored in ./output_data/")