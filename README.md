# CryptoNets-with-Python
In ./SEAL/:

./configure

make


In ./:

g++ -std=c++11 -I SEAL -L SEAL/bin -fPIC -shared wrapper.cpp -o SEAL/libseal.so -lseal

python wrapper.py 


