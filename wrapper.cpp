
// A wrapper for the SEAL library. The wrapper can support basic operations such as
// encoding, decoding and squaring, as mush as initializing variables, genereting new keys
// and freeing memory.


#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <thread>
#include <mutex>
#include <random>
#include <limits>
#include <math.h>
#include "seal/seal.h"
#include <fstream>

using namespace std;
using namespace seal;

// PARAMETERS
EncryptionParameters *parms[5];
std::vector<SmallModulus> q_array;
SEALContext *context[5];
// KEYS
KeyGenerator *keygen[5];
PublicKey public_key[5];
SecretKey secret_key[5];
EvaluationKeys ev_keys[5];
// OBJECTS
Encryptor *encryptor[5];
Decryptor *decryptor[5];
Evaluator *evaluator[5];
PolyCRTBuilder *crtbuilder[5];
int plain_poly_size = 0;
int enc_poly_size = 0;

void deallocate_() {
    for (int i=0; i<5; i++) {
        delete parms[i];
    }
    for (int i=0; i<5; i++) {
        delete context[i];
    }
    for (int i=0; i<5; i++) {
        delete keygen[i];
    }
    for (int i=0; i<5; i++) {
        delete encryptor[i];
    }
    for (int i=0; i<5; i++) {
        delete decryptor[i];
    }
    for (int i=0; i<5; i++) {
        delete evaluator[i];
    }
    for (int i=0; i<5; i++) {
        delete crtbuilder[i];
    }
}

void generate_new_keys_() {
    // PARAMETERS
    q_array = coeff_modulus_128(4096);
    q_array.push_back(small_mods_60bit(63));
    q_array.push_back(small_mods_60bit(42));
    for (int i=0; i<5; i++) {
        parms[i] = new EncryptionParameters;
    }

    // --t
    parms[0]->set_plain_modulus(40961);
    parms[1]->set_plain_modulus(65537);
    parms[2]->set_plain_modulus(114689);
    parms[3]->set_plain_modulus(147457);
    parms[4]->set_plain_modulus(188417);

    std::string file_name = "";
    for (int i=0; i<5; i++) {
        // --n
        parms[i]->set_poly_modulus("1x^4096 + 1");
        // --q
        parms[i]->set_coeff_modulus(q_array);

        context[i] = new SEALContext(*parms[i]);

        // STORE KEYS
        keygen[i] = new KeyGenerator(*context[i]);
        public_key[i] = keygen[i]->public_key();
        secret_key[i] = keygen[i]->secret_key();
        keygen[i]->generate_evaluation_keys(16, ev_keys[i]); // per rilinearizzare dopo lo square
        // --public
        file_name = "./keys/public-" + std::to_string(i);
        std::ofstream pk_stream(file_name, std::ios::out | std::ios::trunc | std::ios::binary);
        public_key[i].save(pk_stream);
        pk_stream.close();
        // --secret
        file_name = "./keys/secret-" + std::to_string(i);
        std::ofstream sk_stream(file_name, std::ios::out | std::ios::trunc | std::ios::binary);
        secret_key[i].save(sk_stream);
        sk_stream.close();
        // evaluation
        file_name = "./keys/evaluation-" + std::to_string(i);
        std::ofstream ek_stream(file_name, std::ios::out | std::ios::trunc | std::ios::binary);
        ev_keys[i].save(ek_stream);
        ek_stream.close();
    }
}

void initialize_() {
    // PARAMETERS
    q_array = coeff_modulus_128(4096);
    q_array.push_back(small_mods_60bit(63));
    q_array.push_back(small_mods_60bit(42));
    for (int i=0; i<5; i++) {
        parms[i] = new EncryptionParameters;
    }

    // --t
    parms[0]->set_plain_modulus(40961);
    parms[1]->set_plain_modulus(65537);
    parms[2]->set_plain_modulus(114689);
    parms[3]->set_plain_modulus(147457);
    parms[4]->set_plain_modulus(188417);

    std::string file_name = ".";
    plain_poly_size = 4096;
    for (int i=0; i<5; i++) {
        // --n
        parms[i]->set_poly_modulus("1x^4096 + 1");
        // --q
        parms[i]->set_coeff_modulus(q_array);

        context[i] = new SEALContext(*parms[i]);

        // LOAD KEYS
        // --public
        file_name = "./keys/public-" + std::to_string(i);
        std::ifstream pk_stream(file_name, std::ios::in | std::ios::binary);
        if (pk_stream) {
            public_key[i].load(pk_stream);
        } else {
            std::cout << "Keys not found" << std::endl;
            std::cout << "Keys not found" << std::endl;
            throw;
        }
        pk_stream.close();
        // --secret
        file_name = "./keys/secret-" + std::to_string(i);
        std::ifstream sk_stream(file_name, std::ios::out | std::ios::binary);
        if (sk_stream) {
            secret_key[i].load(sk_stream);
        } else {
            std::cout << "Keys not found" << std::endl;
            std::cout << "Keys not found" << std::endl;
            throw;
        }
        sk_stream.close();
        // --evaluation
        file_name = "./keys/evaluation-" + std::to_string(i);
        std::ifstream ek_stream(file_name, std::ios::out | std::ios::binary);
        if (ek_stream) {
            ev_keys[i].load(ek_stream);
        } else {
            std::cout << "Keys not found" << std::endl;
            std::cout << "Keys not found" << std::endl;
            throw;
        }
        ek_stream.close();

        // OBJECTS
        encryptor[i] = new Encryptor(*context[i], public_key[i]);
        evaluator[i] = new Evaluator(*context[i]);
        decryptor[i] = new Decryptor(*context[i], secret_key[i]);
        crtbuilder[i] = new PolyCRTBuilder(*context[i]);
    }

    // compute sizes of polynomials
    plain_poly_size = crtbuilder[0]->slot_count();
    enc_poly_size = 2 * q_array.size() * (plain_poly_size + 1);
}

void k_list_(uint64_t * output) {
    for (int i=0; i<5; i++) {
        for (int j=0; j<4; j++) {
            output[(i*4)+j] = evaluator[i]->k_list[j];
        }
    }
}

void encrypt_tensor_(uint64_t *array_input, uint64_t *array_output, int input_axis0_size, int data_size) {
    int poly_groups_count = input_axis0_size / plain_poly_size;
    int last_group_size = input_axis0_size % plain_poly_size;
    int input_index = 0;
    int output_index = 0;
    vector<uint64_t> plain_vector(plain_poly_size, 0);

    for (int poly_group_index=0; poly_group_index<poly_groups_count; poly_group_index++) {
        input_index = poly_group_index * plain_poly_size * data_size * 5;
        output_index = poly_group_index * enc_poly_size * data_size * 5;
        for (int data_index=0; data_index<data_size; data_index++) {
            for (int t_index=0; t_index<5; t_index++) {
                for (int plain_index=0; plain_index<plain_poly_size; plain_index++) {
                    plain_vector[plain_index] = array_input[input_index+(plain_index*data_size*5)];
                }
                Plaintext plain_poly;
                crtbuilder[t_index]->compose(plain_vector, plain_poly);
                Ciphertext encrypted_poly;
                encryptor[t_index]->encrypt(plain_poly, encrypted_poly);
                const uint64_t *encrypted_array = encrypted_poly.pointer();
                for (int enc_index=0; enc_index<enc_poly_size; enc_index++) {
                    array_output[output_index+(enc_index*data_size*5)] = encrypted_array[enc_index];
                }
                input_index++;
                output_index++;
            }
        }
    }

    if (last_group_size!=0) {
        input_index = poly_groups_count * plain_poly_size * data_size * 5;
        output_index = poly_groups_count * enc_poly_size * data_size * 5;
        for (int data_index=0; data_index<data_size; data_index++) {
            for (int t_index=0; t_index<5; t_index++) {
                for (int plain_index=0; plain_index<last_group_size; plain_index++) {
                    plain_vector[plain_index] = array_input[input_index+(plain_index*data_size*5)];
                }
                for (int plain_index=last_group_size; plain_index<plain_poly_size; plain_index++) {
                    plain_vector[plain_index] = 0;
                }
                Plaintext plain_poly;
                crtbuilder[t_index]->compose(plain_vector, plain_poly);
                Ciphertext encrypted_poly;
                encryptor[t_index]->encrypt(plain_poly, encrypted_poly);
                const uint64_t *encrypted_array = encrypted_poly.pointer();
                for (int enc_index=0; enc_index<enc_poly_size; enc_index++) {
                    array_output[output_index+(enc_index*data_size*5)] = encrypted_array[enc_index];
                }
                input_index++;
                output_index++;
            }
        }
    }
}

void decrypt_tensor_(uint64_t *array_input, uint64_t *array_output, int output_axis0_size, int data_size) {
    int poly_groups_count = output_axis0_size / plain_poly_size;
    int last_group_size = output_axis0_size % plain_poly_size;
    int input_index = 0;
    int output_index = 0;
    uint64_t enc_vector[enc_poly_size];

    for (int poly_group_index=0; poly_group_index<poly_groups_count; poly_group_index++) {
        output_index = poly_group_index * plain_poly_size * data_size * 5;
        input_index = poly_group_index * enc_poly_size * data_size * 5;
        for (int data_index=0; data_index<data_size; data_index++) {
            for (int t_index=0; t_index<5; t_index++) {
                for (int enc_index=0; enc_index<enc_poly_size; enc_index++) {
                    enc_vector[enc_index] = array_input[input_index+(enc_index*data_size*5)];
                }
                Ciphertext encrypted_poly(*parms[t_index], 2, enc_vector);
                Plaintext plain_poly;
                decryptor[t_index]->decrypt(encrypted_poly, plain_poly);
                vector<uint64_t> plain_vector_output;
                crtbuilder[t_index]->decompose(plain_poly, plain_vector_output);
                for (int plain_index=0; plain_index<plain_poly_size; plain_index++) {
                    array_output[output_index+(plain_index*data_size*5)] = plain_vector_output[plain_index];
                }                
                input_index++;
                output_index++;
            }
        }
    }

    if (last_group_size!=0) {
        output_index = poly_groups_count * plain_poly_size * data_size * 5;
        input_index = poly_groups_count * enc_poly_size * data_size * 5;
        for (int data_index=0; data_index<data_size; data_index++) {
            for (int t_index=0; t_index<5; t_index++) {
                for (int enc_index=0; enc_index<enc_poly_size; enc_index++) {
                    enc_vector[enc_index] = array_input[input_index+(enc_index*data_size*5)];
                }
                Ciphertext encrypted_poly(*parms[t_index], 2, enc_vector);
                Plaintext plain_poly;
                decryptor[t_index]->decrypt(encrypted_poly, plain_poly);
                vector<uint64_t> plain_vector_output;
                crtbuilder[t_index]->decompose(plain_poly, plain_vector_output);
                for (int plain_index=0; plain_index<last_group_size; plain_index++) {
                    array_output[output_index+(plain_index*data_size*5)] = plain_vector_output[plain_index];
                }                
                input_index++;
                output_index++;
            }
        }
    }
}

void square_tensor_(uint64_t *array_input, uint64_t *array_output, int input_axis0_size, int data_size) {
    int poly_groups_count = input_axis0_size / enc_poly_size;
    int input_index = 0;
    int output_index = 0;
    uint64_t enc_vector[enc_poly_size];

    for (int poly_group_index=0; poly_group_index<poly_groups_count; poly_group_index++) {
        input_index = poly_group_index * enc_poly_size * data_size * 5;
        output_index = poly_group_index * enc_poly_size * data_size * 5;
        for (int data_index=0; data_index<data_size; data_index++) {
            for (int t_index=0; t_index<5; t_index++) {
                for (int enc_index=0; enc_index<enc_poly_size; enc_index++) {
                    enc_vector[enc_index] = array_input[input_index+(enc_index*data_size*5)];
                }
                Ciphertext encrypted_poly(*parms[t_index], 2, enc_vector);
                encrypted_poly.unalias();
                evaluator[t_index]->square(encrypted_poly);
                evaluator[t_index]->relinearize(encrypted_poly, ev_keys[t_index]);
                const uint64_t *encrypted_array = encrypted_poly.pointer();
                for (int enc_index=0; enc_index<enc_poly_size; enc_index++) {
                    array_output[output_index+(enc_index*data_size*5)] = encrypted_array[enc_index];
                }                
                input_index++;
                output_index++;
            }
        }
    }
}

extern "C"
{
    void deallocate() {
        deallocate_();
    }
    void generate_new_keys() {
        generate_new_keys_();
    }
    void initialize() {
        initialize_();
    }
    void k_list(uint64_t * output) {
        k_list_(output);
    }
    void encrypt_tensor(uint64_t *input, uint64_t *output, int input_axis0_size, int data_size) {
        encrypt_tensor_(input, output, input_axis0_size, data_size);
    }
    void decrypt_tensor(uint64_t *input, uint64_t *output, int output_axis0_size, int data_size) {
        decrypt_tensor_(input, output, output_axis0_size, data_size);
    }
    void square_tensor(uint64_t *array_input, uint64_t *array_output, int input_axis0_size, int data_size) {
        square_tensor_(array_input, array_output, input_axis0_size, data_size);
    }

}

/* old code

void print_parameters(const SEALContext &context)
{
    cout << "/ Encryption parameters:" << endl;
    cout << "| poly_modulus: " << context.poly_modulus().to_string() << endl;
    cout << "| poly_modulus.coeff_count: " << context.poly_modulus().coeff_count() << endl;
    cout << "| coeff_modulus size: " << context.total_coeff_modulus().significant_bit_count() << " bits" << endl;
    cout << "| coeff_modulus size: " << context.coeff_modulus().size() << endl;
    cout << "| plain_modulus: " << context.plain_modulus().value() << endl;
    cout << "\\ noise_standard_deviation: " << context.noise_standard_deviation() << endl;
    cout << endl;
}

void stampa_vettore(const vector<uint64_t> &vettore, int limite = -1) {
    int nd0 = 0;
    for (int i = 0; i < vettore.size(); i++)
    {
        if (limite!=-1)
            if (i>limite) {
                cout << "[...] ";
                break;
            }
        if (vettore[i]==0) {
            if (nd0!=-1)
                nd0++;
        } else {
            nd0 = 0;
        }
        if (nd0==3) {
            nd0 = -1;
            cout << "[...] ";
        }
        if (nd0!=-1)
            cout << vettore[i] << " ";
    }
    cout << endl;
}

//----------------------------------------OPERAZIONI----------------------------------------

void somma_tra_polinomi(Ciphertext &encrypted_input1, Ciphertext &encrypted_input2, Decryptor decryptor, EncryptionParameters parms, SEALContext context)
{
    cout << "SOMA TRA DUE POLINOMI:" << endl;
    int size = encrypted_input1.size();
    if (size!=2) {
        cout << "\tERRORE, SI LAVORA SOLO CON SIZE = 2" << endl;
        return;
    }
    const uint64_t *array_input1 = encrypted_input1.pointer();
    const uint64_t *array_input2 = encrypted_input2.pointer();
    int n = encrypted_input1.poly_coeff_count();
    int q_size = encrypted_input1.coeff_mod_count();
    uint64_t q[q_size];
    for (int i=0; i<q_size; i++)
        q[i] = parms.coeff_modulus()[i].value();
    uint64_t array_output[n * q_size * size];

    PolyCRTBuilder crtbuilder(context);
    Plaintext plain_input1;
    decryptor.decrypt(encrypted_input1, plain_input1);
    vector<uint64_t> plain_vector1;
    crtbuilder.decompose(plain_input1, plain_vector1);
    Plaintext plain_input2;
    decryptor.decrypt(encrypted_input2, plain_input2);
    vector<uint64_t> plain_vector2;
    crtbuilder.decompose(plain_input2, plain_vector2);
    cout << "\t" << "Polinomio 1: ";
    stampa_vettore(plain_vector1);
    cout << "\t" << "Polinomio 2: ";
    stampa_vettore(plain_vector2);
    cout << "\t" << "budget 1 prima: " << decryptor.invariant_noise_budget(encrypted_input1) << " bits" << endl;
    cout << "\t" << "budget 2 prima: " << decryptor.invariant_noise_budget(encrypted_input2) << " bits" << endl;

    int index = -1;
    for (int index_on_size=0; index_on_size<2; index_on_size++){
        for (int index_on_q_size=0; index_on_q_size<q_size; index_on_q_size++){
            for (int index_on_n=0; index_on_n<n; index_on_n++){
                index = index_on_size * q_size * n;
                index += index_on_q_size * n;
                index += index_on_n;
                array_output[index] = (array_input1[index] + array_input2[index]) % q[index_on_q_size];
            }
        }
    }

    Ciphertext encrypted_output(parms, 2, array_output);
    Plaintext plain_output;
    decryptor.decrypt(encrypted_output, plain_output);
    vector<uint64_t> plain_vector_output;
    crtbuilder.decompose(plain_output, plain_vector_output);
    cout << "\t" << "Risultato: ";
    stampa_vettore(plain_vector_output);
    cout << "\t" << "budget dopo: " << decryptor.invariant_noise_budget(encrypted_output) << " bits" << endl;
}

void prodotto_polinomio_scalare(Ciphertext &encrypted_input, int scalar, Decryptor decryptor, EncryptionParameters parms, SEALContext context)
{
    cout << "PRODOTTO TRA UN POLINOMIO E UNO SCALARE:" << endl;
    int size = encrypted_input.size();
    if (size!=2) {
        cout << "\tERRORE, SI LAVORA SOLO CON SIZE = 2" << endl;
        return;
    }
    const uint64_t *array_input = encrypted_input.pointer();
    int n = encrypted_input.poly_coeff_count();
    int q_size = encrypted_input.coeff_mod_count();
    uint64_t q[q_size];
    for (int i=0; i<q_size; i++)
        q[i] = parms.coeff_modulus()[i].value();
    uint64_t array_output[n * q_size * size];

    PolyCRTBuilder crtbuilder(context);
    Plaintext plain_input;
    decryptor.decrypt(encrypted_input, plain_input);
    vector<uint64_t> plain_vector;
    crtbuilder.decompose(plain_input, plain_vector);
    cout << "\t" << "Polinomio: ";
    stampa_vettore(plain_vector);
    cout << "\t" << "Scalare: " << scalar << endl;
    cout << "\t" << "budget prima: " << decryptor.invariant_noise_budget(encrypted_input) << " bits" << endl;

    int index = -1;
    for (int index_on_size=0; index_on_size<2; index_on_size++){
        for (int index_on_q_size=0; index_on_q_size<q_size; index_on_q_size++){
            for (int index_on_n=0; index_on_n<n; index_on_n++){
                index = index_on_size * q_size * n;
                index += index_on_q_size * n;
                index += index_on_n;
                array_output[index] = (array_input[index] * scalar) % q[index_on_q_size];
            }
        }
    }

    Ciphertext encrypted_output(parms, 2, array_output);
    Plaintext plain_output;
    decryptor.decrypt(encrypted_output, plain_output);
    vector<uint64_t> plain_vector_output;
    crtbuilder.decompose(plain_output, plain_vector_output);
    cout << "\t" << "Risultato: ";
    stampa_vettore(plain_vector_output);
    cout << "\t" << "budget dopo: " << decryptor.invariant_noise_budget(encrypted_output) << " bits" << endl;
}

void somma_polinomio_scalare(Ciphertext &encrypted_input, int scalar, uint64_t *k, int limit, Decryptor decryptor, EncryptionParameters parms, SEALContext context)
{
    cout << "SOMMA TRA UN POLINOMIO E UNO SCALARE:" << endl;
    int size = encrypted_input.size();
    if (size!=2) {
        cout << "\tERRORE, SI LAVORA SOLO CON SIZE = 2" << endl;
        return;
    }
    const uint64_t *array_input = encrypted_input.pointer();
    int n = encrypted_input.poly_coeff_count();
    int q_size = encrypted_input.coeff_mod_count();
    uint64_t q[q_size];
    for (int i=0; i<q_size; i++)
        q[i] = parms.coeff_modulus()[i].value();
    uint64_t array_output[n * q_size * size];

    PolyCRTBuilder crtbuilder(context);
    Plaintext plain_input;
    decryptor.decrypt(encrypted_input, plain_input);
    vector<uint64_t> plain_vector;
    crtbuilder.decompose(plain_input, plain_vector);
    cout << "\t" << "Polinomio: ";
    stampa_vettore(plain_vector);
    cout << "\t" << "Scalare: " << scalar << endl;
    cout << "\t" << "budget prima: " << decryptor.invariant_noise_budget(encrypted_input) << " bits" << endl;

    int index = -1;
    uint64_t temp = -1;
    for (int index_on_q_size=0; index_on_q_size<q_size; index_on_q_size++){
        for (int index_on_n=0; index_on_n<n; index_on_n++){
            index = index_on_q_size * n;
            index += index_on_n;
            temp = (scalar * k[index_on_q_size]) % q[index_on_q_size];
            if (index_on_n>0) temp = 0;
            array_output[index] = (array_input[index] + temp) % q[index_on_q_size];
        }
    }

    for (int index_on_q_size=0; index_on_q_size<q_size; index_on_q_size++){
        for (int index_on_n=0; index_on_n<n; index_on_n++){
            index = q_size * n;
            index += index_on_q_size * n;
            index += index_on_n;
            array_output[index] = array_input[index];
        }
    }

    Ciphertext encrypted_output(parms, 2, array_output);
    Plaintext plain_output;
    decryptor.decrypt(encrypted_output, plain_output);
    vector<uint64_t> plain_vector_output;
    crtbuilder.decompose(plain_output, plain_vector_output);
    cout << "\t" << "Risultato: ";
    stampa_vettore(plain_vector_output, 7);
    cout << "\t" << "budget dopo: " << decryptor.invariant_noise_budget(encrypted_output) << " bits" << endl;
}

void quadrato_dei_coefficienti(Ciphertext &encrypted_input, Decryptor decryptor, EncryptionParameters parms, EvaluationKeys ev_keys, Evaluator evaluator, SEALContext context)
{
    cout << "ELEVAZIONE AL QUADRATO DI UN POLINOMIO:" << endl;
    int size = encrypted_input.size();
    if (size!=2) {
        cout << "\tERRORE, SI LAVORA SOLO CON SIZE = 2" << endl;
        return;
    }

    PolyCRTBuilder crtbuilder(context);
    Plaintext plain_input;
    decryptor.decrypt(encrypted_input, plain_input);
    vector<uint64_t> plain_vector;
    crtbuilder.decompose(plain_input, plain_vector);
    cout << "\t" << "Polinomio: ";
    stampa_vettore(plain_vector);
    cout << "\t" << "budget prima: " << decryptor.invariant_noise_budget(encrypted_input) << " bits" << endl;
    cout << "\t" << "Size prima: " << size << endl;

    evaluator.square(encrypted_input);
    cout << "\t" << "Size dopo square: " << encrypted_input.size() << endl;
    cout << "\t" << "budget dopo square: " << decryptor.invariant_noise_budget(encrypted_input) << " bits" << endl;

    evaluator.relinearize(encrypted_input, ev_keys);
    int budget = decryptor.invariant_noise_budget(encrypted_input);
    cout << "\t" << "Size dopo relienarize: " << encrypted_input.size() << endl;
    cout << "\t" << "budget dopo relienarize: " << budget << " bits" << endl;

    if (budget==0) {
        cout << "\tbudget esaurito, terminazione anticipata" << endl;
        return;
    }

    Plaintext plain_output;
    decryptor.decrypt(encrypted_input, plain_output);
    vector<uint64_t> plain_vector_output;
    crtbuilder.decompose(plain_output, plain_vector_output);
    cout << "\t" << "Risultato: ";
    stampa_vettore(plain_vector_output);
}

*/