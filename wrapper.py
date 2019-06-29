import ctypes
import numpy as np

lib = ctypes.cdll.LoadLibrary('./SEAL/libseal.so')
lib.initialize()
#lib.encrypt_array.restype = ctypes.POINTER(ctypes.c_uint64)
lib.square_tensor.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.encrypt_tensor.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.decrypt_tensor.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.k_list.argtypes = [ctypes.POINTER(ctypes.c_uint64)]

#slib = CDLL("slib.dll")
#slib.print_struct.argtypes = [POINTER(TestStruct)]
#slib.print_struct.restype = None


#test e info
#lib.prova()
#lib.q_list()
#36028797014376449
#36028797013327873
#1152921504241942529
#1152921504369344513


q_list = [36028797014376449, 36028797013327873, 1152921504241942529, 1152921504369344513]

#k_list = [3578163283476560, 5902913912171171, 256727058426082317, 114641959114678357]
#per t=40961
#k_list = [3578163283476560, 5902913912171171, 256727058426082317, 114641959114678357]
#per t=188417
#k_list = [23809558905221309, 3180918061091669, 313592054702576267, 922602767607469907]

temp = (ctypes.c_uint64 * 20)()
lib.k_list(temp)
k_list = []
for t in range(5):
	temp2 = []
	for k in range(4):
		temp2.append(int(temp[(t*4)+k]))
	k_list.append(temp2)
temp = None
temp2 = None

#lib.prova()

def cripta(input_array):
	grado = 4096
	shape = input_array.shape
	shape_size = len(shape)
	q = 4
	assert shape[-1] == 5, "Errore in cripta: input_array deve avere size[-1] = 5"

	#calcolo dimensioni
	input_size = 1
	output_size = 1
	input_axis0_size = shape[0]
	output_axis0_size = 1
	data_size = 1

	for i in range(shape_size):
		input_size = input_size * shape[i]
	data_size = input_size // input_axis0_size
	divisione = input_axis0_size//grado
	resto = input_axis0_size%grado
	output_axis0_size = divisione * ((grado+1)*2*q)
	if (resto!=0):
		output_axis0_size = output_axis0_size + ((grado+1)*2*q)
	output_size = output_axis0_size * data_size

	#chiamata funzione
	input_array.shape = (input_size,)
	input_vec = (ctypes.c_uint64 * (input_size))()
	for i in range(input_size):
		input_vec[i] = input_array[i]
	output_vec = (ctypes.c_uint64 * (output_size))()
	data_size = data_size // 5 #adegua data size allo standard del wrap in c++
	lib.encrypt_tensor(input_vec, output_vec, input_axis0_size, output_axis0_size, data_size)

	#costruzione risultato
	output_array = np.empty((output_size), dtype=np.uint64)
	for i in range(output_size):
		output_array[i] = output_vec[i]
	shape = (output_axis0_size,) + shape[1:] #output shape
	output_array.shape = shape
	input_vec = None
	output_vec = None
	input_array = None
	return output_array

def decripta(input_array, output_axis0_size):
	grado = 4096
	shape = input_array.shape
	shape_size = len(shape)
	q = 4
	assert (shape[0]%((grado+1)*4*2)) == 0, "Errore in decripta: input_array ha size[0] inaspettato"
	assert shape[-1] == 5, "Errore in decripta: input_array deve avere size[-1] = 5"

	#calcolo dimensioni
	input_size = 1
	output_size = 1
	input_axis0_size = shape[0]
	output_axis0_size = output_axis0_size
	data_size = 1

	for i in range(shape_size):
		input_size = input_size * shape[i]
	data_size = input_size // input_axis0_size
	output_size = output_axis0_size * data_size

	#chiamata funzione
	input_array.shape = (input_size,)
	input_vec = (ctypes.c_uint64 * (input_size))()
	for i in range(input_size):
		input_vec[i] = input_array[i]
	output_vec = (ctypes.c_uint64 * (output_size))()
	data_size = data_size // 5 #adegua data size allo standard del wrap in c++
	lib.decrypt_tensor(input_vec, output_vec, input_axis0_size, output_axis0_size, data_size)

	#costruzione risultato
	output_array = np.empty((output_size), dtype=np.uint64)
	for i in range(output_size):
		output_array[i] = output_vec[i]
	shape = (output_axis0_size,) + shape[1:] #output shape
	output_array.shape = shape
	input_vec = None
	output_vec = None
	input_array = None
	return output_array


def alquadrato(input_array):
	grado = 4096
	shape = input_array.shape
	shape_size = len(shape)
	q = 4
	assert (shape[0]%((grado+1)*4*2)) == 0, "Errore in decripta: input_array ha size[0] inaspettato"
	assert shape[-1] == 5, "Errore in decripta: input_array deve avere size[-1] = 5"

	#calcolo dimensioni
	input_size = 1
	input_axis0_size = shape[0]
	data_size = 1

	for i in range(shape_size):
		input_size = input_size * shape[i]
	data_size = input_size // input_axis0_size
	output_size = input_size
	output_axis0_size = input_axis0_size

	#chiamata funzione
	input_array.shape = (input_size,)
	input_vec = (ctypes.c_uint64 * (input_size))()
	for i in range(input_size):
		input_vec[i] = input_array[i]
	output_vec = (ctypes.c_uint64 * (output_size))()
	data_size = data_size // 5 #adegua data size allo standard del wrap in c++
	lib.square_tensor(input_vec, output_vec, input_axis0_size, output_axis0_size, data_size)

	#costruzione risultato
	output_array = np.empty((output_size), dtype=np.uint64)
	for i in range(output_size):
		output_array[i] = output_vec[i]
	#shape = (output_axis0_size,) + shape[1:] #output shape
	output_array.shape = shape
	input_vec = None
	output_vec = None
	input_array = None
	return output_array


def flatten(tensore):
	dimensione = tensore.size
	tensore.shape = (dimensione,)


def confronta(tensoreA, tensoreB):
	print("")
	if (tensoreA.size!=tensoreB.size):
		print("I due tensori hanno dimensioni diverse:")
		print(tensoreA.shape)
		print(tensoreB.shape)
		return
	flatten(tensoreA)
	flatten(tensoreB)
	confronto = (tensoreA==tensoreB)
	for i in range(tensoreB.size):
		if (not confronto[i]):
			print("Differenza in i =", i)
			print(tensoreA[i])
			print(tensoreB[i])
			return
	print("Sono uguali!")


a = np.arange(120)
a.shape = (4,2,3,5)


if (False):
	b = cripta(a)
	a = None
	gruppi_di_polinomi = 4097*4*2
	gruppi_di_polinomi = b.shape[0]//gruppi_di_polinomi
	for axis1 in range(b.shape[1]):
		for axis2 in range(b.shape[2]):
			for axis3 in range(b.shape[3]):
				for gdp in range(gruppi_di_polinomi):
					for q_index in range(4):
						index = gdp*4097*4*2 + (4097*q_index)
						temp = b[index,axis1,axis2,axis3].item()
						temp = temp + 12*k_list[axis3][q_index]
						temp = temp % q_list[q_index]
						b[index,axis1,axis2,axis3] = temp

	c = decripta(b, 4) #4 HARD CODED
	b = None
	c.shape = (120,)
	print(c)
else:
	a = None


print("")

ris = np.zeros((2,2), dtype=object)
t_moduli = [40961, 65537, 114689, 147457, 188417]
gruppi_di_polinomi = 4097*4*2
gruppi_di_polinomi = ris.shape[0]//gruppi_di_polinomi

a = np.arange(20)
a.shape = (2,2,5)

print(a)

a = cripta(a)

a = alquadrato(a)

a = decripta(a, 2)
print(a)



#confronta(ris, temp)

'''

ris = np.load("./matrices/5_1_dense2_bias.npy")
temp = cripta(ris)
temp = decripta(temp, 10000)

#    for i in range(10000):
#        for j in range(10):
#            for k in range(5):
#                if (temp[i,j,k] == 69205):
#                    print("indice ",i,j,k)
#                    lib.deallocate()
#                    exit()

#temp.shape = (500000,)
#temp.sort()
#ris.sort()


confronta(ris, temp)


temp = cripta(a)
temp = decripta(temp, 4)
confronta(a, temp)


a.shape = (4,2,3,5)
for i1 in range(4):
	for i2 in range(2):
		for i3 in range(3):
			for i4 in range(5):
				a[i1,i2,i3,i4] = ris[i4 + (i3*5) + (i2*15) + (i1*30)]

temp = cripta(a)
temp = decripta(temp, 4)

confronta(a, temp)

#157070
'''
#a = a + 40845
#40960
#print(a[:,:,:,0])
#temp = cripta(a)
#temp = decripta(temp, 4)

#confronta(a, temp)




'''
dimensione_array_criptati = 4*2*4097

### VETTORE 1
vettore_plain_1 = (ctypes.c_uint64 * 4096)()
for i in range(4069):
	if i>7:
		vettore_plain_1[i] = 0
	else:
		vettore_plain_1[i] = i

vettore_cifrato_1 = lib.encrypt_array(vettore_plain_1)
vettore_cifrato_1_copiato = np.empty((dimensione_array_criptati,), dtype=np.uint64)
for i in range(dimensione_array_criptati):
	vettore_cifrato_1_copiato[i] = vettore_cifrato_1[i]

vettore_cifrato_1b = lib.square(vettore_cifrato_1)

lib.decrypt_array(vettore_cifrato_1b)

### VETTORE 2
vettore_plain_2 = (ctypes.c_uint64 * 4096)()
for i in range(4069):
		vettore_plain_2[i] = 0
vettore_plain_2[0] = 3
vettore_plain_2[1] = 2
vettore_plain_2[5] = 1
vettore_plain_2[6] = 1

vettore_cifrato_2 = lib.encrypt_array(vettore_plain_2)



## CALCOLI
q = [36028797014376449, 36028797013327873, 1152921504241942529, 1152921504369344513]

index = 0
vettore_cifrato_3 = (ctypes.c_uint64 * dimensione_array_criptati)()
for index_on_size in range(2):
	for index_on_q_size in range(4):
		for index_on_n in range(4097):
				index = index_on_size * 4 * 4097;
				index += index_on_q_size * 4097;
				index += index_on_n;
				vettore_cifrato_3[index] = (vettore_cifrato_1[index] * 4) % q[index_on_q_size]

lib.decrypt_array(vettore_cifrato_3)


index = 0
for index_on_size in range(2):
	for index_on_q_size in range(4):
		for index_on_n in range(4097):
				index = index_on_size * 4 * 4097;
				index += index_on_q_size * 4097;
				index += index_on_n;
				vettore_cifrato_3[index] = (vettore_cifrato_1_copiato[index].item() * 2) % q[index_on_q_size]



lib.decrypt_array(vettore_cifrato_3)


'''
print("")
lib.deallocate()