# CryptoNets-with-Python

Welcome! This project aims to replicate the work done in this [paper](http://proceedings.mlr.press/v48/gilad-bachrach16.pdf).

## Requirement:

* TensorFlow 1.8.0
* Python 3
* A C++11 compiler

## Compiling instruction

To compile the project, enter the SEAL folder, generate the Makefile and compile the seal library:

```bash
cd ./SEAL/
./configure
make
```

Then, in the parent folder, compile the wrapper.cpp file like this:

```bash
cd ..
g++ -std=c++11 -I SEAL -L SEAL/bin -fPIC -shared wrapper.cpp -o SEAL/libseal.so -lseal
```

## Running the code

The project consists of 5 files: ```train.py```, ```pre_encode.py```, ```infere_plain.py```, ```infere\_enc.py``` and ```post\_decode.py```. They must be executed sequentially, refer to page 15 of the report for more details.

> Attention! The execution of these files, especially ```infere\_enc.py```, can take a long time (even beyond 10 hours), depending on the machine used.

## Genereting a new set of keys

The project is already supplied with a set of keys. If you want to generate a new set of keys you can do it using the API provided by the file ```wrapper.py``` in this way:

```bash
python3
from wrapper import SEAL
SEALobj = SEAL()
SEALobj.generate\_new\_keys()
```

