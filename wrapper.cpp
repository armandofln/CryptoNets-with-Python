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

using namespace std;
using namespace seal;

//----------------------------------------STAMPE----------------------------------------

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
    cout << "\t" << "Mana 1 prima: " << decryptor.invariant_noise_budget(encrypted_input1) << " bits" << endl;
    cout << "\t" << "Mana 2 prima: " << decryptor.invariant_noise_budget(encrypted_input2) << " bits" << endl;

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
    cout << "\t" << "Mana dopo: " << decryptor.invariant_noise_budget(encrypted_output) << " bits" << endl;
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
    cout << "\t" << "Mana prima: " << decryptor.invariant_noise_budget(encrypted_input) << " bits" << endl;

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
    cout << "\t" << "Mana dopo: " << decryptor.invariant_noise_budget(encrypted_output) << " bits" << endl;
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
    cout << "\t" << "Mana prima: " << decryptor.invariant_noise_budget(encrypted_input) << " bits" << endl;

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
    cout << "\t" << "Mana dopo: " << decryptor.invariant_noise_budget(encrypted_output) << " bits" << endl;
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
    cout << "\t" << "Mana prima: " << decryptor.invariant_noise_budget(encrypted_input) << " bits" << endl;
    cout << "\t" << "Size prima: " << size << endl;

    evaluator.square(encrypted_input);
    cout << "\t" << "Size dopo square: " << encrypted_input.size() << endl;
    cout << "\t" << "Mana dopo square: " << decryptor.invariant_noise_budget(encrypted_input) << " bits" << endl;

    evaluator.relinearize(encrypted_input, ev_keys);
    int mana = decryptor.invariant_noise_budget(encrypted_input);
    cout << "\t" << "Size dopo relienarize: " << encrypted_input.size() << endl;
    cout << "\t" << "Mana dopo relienarize: " << mana << " bits" << endl;

    if (mana==0) {
        cout << "\tMana esaurito, terminazione anticipata" << endl;
        return;
    }

    Plaintext plain_output;
    decryptor.decrypt(encrypted_input, plain_output);
    vector<uint64_t> plain_vector_output;
    crtbuilder.decompose(plain_output, plain_vector_output);
    cout << "\t" << "Risultato: ";
    stampa_vettore(plain_vector_output);
}

//----------------------------------------LIB----------------------------------------

EncryptionParameters *parms[5];
std::vector<SmallModulus> q_array;
SEALContext *context[5];
//
KeyGenerator *keygen[5];
PublicKey public_key[5];
SecretKey secret_key[5];
EvaluationKeys ev_keys[5];
//
Encryptor *encryptor[5];
Decryptor *decryptor[5];
Evaluator *evaluator[5];
PolyCRTBuilder *crtbuilder[5];
int slot_count = 0;


void deallocate_() {
    cout << "Deallocazione in corso..." << endl;
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
    cout << "Fatto!" << endl;
}

void prova_() {
    cout << "slot_count: " << slot_count << endl;
    for (int i=0; i<4; i++) {
        cout << "q_" << i << " = " << parms[0]->coeff_modulus()[i].value() << endl;
        cout << "k_" << i << " = " << evaluator[0]->k_list[i] << endl;
        cout << endl;
    }
}

void k_list_(uint64_t * output) {
    for (int i=0; i<5; i++) {
        for (int j=0; j<4; j++) {
            output[(i*4)+j] = evaluator[i]->k_list[j];
        }
    }
}

void initialize_() {
    cout << "Inizializzazione in corso..." << endl;
    // PARAMETRI
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

    for (int i=0; i<5; i++) {
        // --n
        parms[i]->set_poly_modulus("1x^4096 + 1");
        // --q
        parms[i]->set_coeff_modulus(q_array);

        context[i] = new SEALContext(*parms[i]);

        // CHIAVI
        keygen[i] = new KeyGenerator(*context[i]);
        public_key[i] = keygen[i]->public_key();
        secret_key[i] = keygen[i]->secret_key();
        keygen[i]->generate_evaluation_keys(16, ev_keys[i]); // per rilinearizzare dopo lo square

        // OGGETTI
        encryptor[i] = new Encryptor(*context[i], public_key[i]);
        evaluator[i] = new Evaluator(*context[i]);
        decryptor[i] = new Decryptor(*context[i], secret_key[i]);
        crtbuilder[i] = new PolyCRTBuilder(*context[i]);
    }

    // inizializza slot count
    slot_count = crtbuilder[0]->slot_count();
    cout << "Fatto!" << endl;
}

const uint64_t * encrypt_array_(uint64_t *vettore) {
    vector<uint64_t> pod_matrix(vettore, vettore + slot_count);
    cout << "Cifrazione del vettore:" << endl;
    stampa_vettore(pod_matrix);
    Plaintext plain_matrix;
    crtbuilder[0]->compose(pod_matrix, plain_matrix);
    Ciphertext encrypted_matrix;
    encryptor[0]->encrypt(plain_matrix, encrypted_matrix);
    cout << "Fatto" << endl;
    //per forza const:
    const uint64_t *puntatore = encrypted_matrix.pointer();
    return puntatore;
}

void decrypt_array_(uint64_t *vettore) {
    Ciphertext encrypted_output(*parms[0], 2, vettore);
    Plaintext plain_output;
    decryptor[0]->decrypt(encrypted_output, plain_output);
    vector<uint64_t> plain_vector_output;
    crtbuilder[0]->decompose(plain_output, plain_vector_output);
    cout << "Decriptazione: ";
    stampa_vettore(plain_vector_output, 12);
    cout << "Mana dopo: " << decryptor[0]->invariant_noise_budget(encrypted_output) << " bits" << endl;
}

const uint64_t * square_(uint64_t *vettore) {
    Ciphertext encrypted_input(*parms[0], 2, vettore);
    encrypted_input.unalias();
    evaluator[0]->square(encrypted_input);
    evaluator[0]->relinearize(encrypted_input, ev_keys[0]);
    const uint64_t *puntatore = encrypted_input.pointer();
    return puntatore;
}

void q_list_() {
    /*
    int n = encrypted_matrix.poly_coeff_count();
    int size = encrypted_matrix.size();
    int numero_di_q = encrypted_matrix.coeff_mod_count();
    uint64_t t = parms[0].plain_modulus().value();
    uint64_t k[numero_di_q];
    cout << "size = " << size << endl;
    cout << "numero di q = " << numero_di_q << endl;
    cout << "n = " << n << endl;
    cout << "t = " << t << endl;
    for (int i=0; i<numero_di_q; i++) {
        k[i] = evaluator.k_parameters_list[i];
        cout << "q_" << i << " = " << parms[0].coeff_modulus()[i].value() << endl;
        cout << "k_" << i << " = " << k[i] << endl;*/
    int numero_di_q = 4;
    uint64_t q_list[numero_di_q];
    for (int i=0; i<numero_di_q; i++) {
        cout << parms[0]->coeff_modulus()[i].value() << endl;
    }
    //uint64_t *puntatore = &(q_array[0]);
    //return q_list;
}


void encrypt_tensor_(uint64_t *array_input, uint64_t *array_output, int input_axis0_size, int output_axis0_size, int data_size) {
    int grado = 4096;
    int divisione = input_axis0_size / grado;
    int resto = input_axis0_size % grado;
    int input_index = 0;
    int output_index = 0;

    vector<uint64_t> temp(slot_count, 0);

    for (int poly_group_index=0; poly_group_index<divisione; poly_group_index++) {
        input_index = poly_group_index * grado * data_size * 5;
        output_index = poly_group_index * (4097 * 4 * 2) * data_size * 5;
        for (int k=0; k<data_size; k++) {
            for (int i=0; i<5; i++) {
                for (int j=0; j<grado; j++) {
                    temp[j] = array_input[input_index+(j*data_size*5)];
                }

                Plaintext plain_poly;
                crtbuilder[i]->compose(temp, plain_poly);
                Ciphertext encrypted_poly;
                encryptor[i]->encrypt(plain_poly, encrypted_poly);
                const uint64_t *encrypted_array = encrypted_poly.pointer();

                for (int j=0; j<(4097 * 4 * 2); j++) {
                    array_output[output_index+(j*data_size*5)] = encrypted_array[j];
                }                
                
                input_index++;
                output_index++;
            }
        }
    }

    if (resto!=0) {
        input_index = divisione * grado * data_size * 5;
        output_index = divisione * (4097 * 4 * 2 )* data_size * 5;
        for (int k=0; k<data_size; k++) {
            for (int i=0; i<5; i++) {
                for (int j=0; j<resto; j++) {
                    temp[j] = array_input[input_index+(j*data_size*5)];
                }
                for (int j=resto; j<grado; j++) {
                    temp[j] = 0;
                }

                Plaintext plain_poly;
                crtbuilder[i]->compose(temp, plain_poly);
                Ciphertext encrypted_poly;
                encryptor[i]->encrypt(plain_poly, encrypted_poly);
                const uint64_t *encrypted_array = encrypted_poly.pointer();

                for (int j=0; j<(4097 * 4 * 2); j++) {
                    array_output[output_index+(j*data_size*5)] = encrypted_array[j];
                }                
                
                input_index++;
                output_index++;
            }
        }
    }

}


void decrypt_tensor_(uint64_t *array_input, uint64_t *array_output, int input_axis0_size, int output_axis0_size, int data_size) {
    int grado = 4096;
    int divisione = output_axis0_size / grado;
    int resto = output_axis0_size % grado;
    int input_index = 0;
    int output_index = 0;

    uint64_t temp[4097 * 4 * 2];

    for (int poly_group_index=0; poly_group_index<divisione; poly_group_index++) {
        output_index = poly_group_index * grado * data_size*5;
        input_index = poly_group_index * (4097 * 4 * 2) * data_size * 5;
        for (int k=0; k<data_size; k++) {
            for (int i=0; i<5; i++) {
                for (int j=0; j<(4097 * 4 * 2); j++) {
                    temp[j] = array_input[input_index+(j*data_size*5)];
                }

                Ciphertext encrypted_poly(*parms[i], 2, temp);
                Plaintext plain_poly;
                decryptor[i]->decrypt(encrypted_poly, plain_poly);
                vector<uint64_t> plain_vector_output;
                crtbuilder[i]->decompose(plain_poly, plain_vector_output);

                for (int j=0; j<grado; j++) {
                    array_output[output_index+(j*data_size*5)] = plain_vector_output[j];
                }                
                
                input_index++;
                output_index++;
            }
        }
    }

    if (resto!=0) {
        output_index = divisione * grado * data_size * 5;
        input_index = divisione * (4097 * 4 * 2) * data_size * 5;
        for (int k=0; k<data_size; k++) {
            for (int i=0; i<5; i++) {
                for (int j=0; j<(4097 * 4 * 2); j++) {
                    temp[j] = array_input[input_index+(j*data_size*5)];
                }

                Ciphertext encrypted_poly(*parms[i], 2, temp);
                Plaintext plain_poly;
                decryptor[i]->decrypt(encrypted_poly, plain_poly);
                vector<uint64_t> plain_vector_output;
                crtbuilder[i]->decompose(plain_poly, plain_vector_output);

                for (int j=0; j<resto; j++) {
                    array_output[output_index+(j*data_size*5)] = plain_vector_output[j];
                }                
                
                input_index++;
                output_index++;
            }
        }
    }

}


void square_tensor_(uint64_t *array_input, uint64_t *array_output, int input_axis0_size, int output_axis0_size, int data_size) {
    int grado = 4096;
    int divisione = output_axis0_size / (4097 * 4 * 2);
    int input_index = 0;
    int output_index = 0;
    int inutile = -1;
    uint64_t temp[4097 * 4 * 2];

    for (int poly_group_index=0; poly_group_index<divisione; poly_group_index++) {
        input_index = poly_group_index * (4097 * 4 * 2) * data_size * 5;
        output_index = poly_group_index * (4097 * 4 * 2) * data_size * 5;
        for (int k=0; k<data_size; k++) {
            for (int i=0; i<5; i++) {

                for (int j=0; j<(4097 * 4 * 2); j++) {
                    temp[j] = array_input[input_index+(j*data_size*5)];
                }

                Ciphertext encrypted_poly(*parms[i], 2, temp);
                encrypted_poly.unalias();
                evaluator[i]->square(encrypted_poly);
                evaluator[i]->relinearize(encrypted_poly, ev_keys[i]);
                const uint64_t *puntatore = encrypted_poly.pointer();

                for (int j=0; j<(4097 * 4 * 2); j++) {
                    array_output[output_index+(j*data_size*5)] = puntatore[j];
                }                
             
                input_index++;
                output_index++;
            }
        }
    }

}

extern "C"
{
    void prova() {
        prova_();
    }
    void deallocate() {
        deallocate_();
    }
    void initialize() {
        initialize_();
    }
    const uint64_t * encrypt_array(uint64_t *vettore) {
        return encrypt_array_(vettore);
    }
    void decrypt_array(uint64_t *vettore) {
        decrypt_array_(vettore);
    }
    void q_list() {
        q_list_();
    }
    const uint64_t * square(uint64_t *vettore) {
        return square_(vettore);
    }
    void encrypt_tensor(uint64_t *input, uint64_t *output, int input_axis0_size, int output_axis0_size, int data_size) {
        encrypt_tensor_(input, output, input_axis0_size, output_axis0_size, data_size);
    }
    void decrypt_tensor(uint64_t *input, uint64_t *output, int input_axis0_size, int output_axis0_size, int data_size) {
        decrypt_tensor_(input, output, input_axis0_size, output_axis0_size, data_size);
    }
    void k_list(uint64_t * output) {
        k_list_(output);
    }
    void square_tensor(uint64_t *array_input, uint64_t *array_output, int input_axis0_size, int output_axis0_size, int data_size) {
        square_tensor_(array_input, array_output, input_axis0_size, output_axis0_size, data_size);
    }
}

//----------------------------------------MAIN----------------------------------------

int main()
{
    /*
    // SET
    // -parametri
    EncryptionParameters parms;
    // --n
    parms.set_poly_modulus("1x^4096 + 1");
    // --q
    std::vector<SmallModulus> vettore_q = coeff_modulus_128(4096);
    vettore_q.push_back(small_mods_60bit(63));
    vettore_q.push_back(small_mods_60bit(42));
    parms.set_coeff_modulus(vettore_q);
    // --t
    parms.set_plain_modulus(40961);
 
    SEALContext context(parms);
    print_parameters(context);

    // CHIAVI
    KeyGenerator keygen(context);
    PublicKey public_key = keygen.public_key();
    SecretKey secret_key = keygen.secret_key();
    EvaluationKeys ev_keys;
    keygen.generate_evaluation_keys(16, ev_keys); // per rinealizzare dopo lo square

    // CREAZIONE OGGETTI
    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key);
    PolyCRTBuilder crtbuilder(context);
    //-------------------------------------------------------------------------------------------------------------------------------
    // POLINOMIO 1
    int slot_count = crtbuilder.slot_count();
    int row_size = slot_count / 2;
    vector<uint64_t> pod_matrix(slot_count, 0);
    pod_matrix[0] = 0;
    pod_matrix[1] = 1;
    pod_matrix[2] = 2;
    pod_matrix[3] = 3;
    pod_matrix[row_size] = 4;
    pod_matrix[row_size + 1] = 5;
    pod_matrix[row_size + 2] = 6;
    pod_matrix[row_size + 3] = 7;
    cout << "Primo vettore plaintext:" << endl;
    stampa_vettore(pod_matrix);
    Plaintext plain_matrix;
    crtbuilder.compose(pod_matrix, plain_matrix);
    Ciphertext encrypted_matrix;
    encryptor.encrypt(plain_matrix, encrypted_matrix);

    // POLINOMIO 2
    vector<uint64_t> pod_matrix2(slot_count, 0);
    for (int i = 0; i < 10; i++) //10
    {
        pod_matrix2[i] = ((i % 2) + 1);
    }
    cout << "Secondo vettore plaintext:" << endl;
    stampa_vettore(pod_matrix2);
    Plaintext plain_matrix2;
    crtbuilder.compose(pod_matrix2, plain_matrix2);
    Ciphertext encrypted_matrix2;
    encryptor.encrypt(plain_matrix2, encrypted_matrix2);
    //-------------------------------------------------------------------------------------------------------------------------------
    // PARAMETRI
    int n = encrypted_matrix.poly_coeff_count();
    int size = encrypted_matrix.size();
    int numero_di_q = encrypted_matrix.coeff_mod_count();
    uint64_t t = parms.plain_modulus().value();
    uint64_t k[numero_di_q];
    cout << "size = " << size << endl;
    cout << "numero di q = " << numero_di_q << endl;
    cout << "n = " << n << endl;
    cout << "t = " << t << endl;
    for (int i=0; i<numero_di_q; i++) {
        k[i] = evaluator.k_parameters_list[i];
        cout << "q_" << i << " = " << parms.coeff_modulus()[i].value() << endl;
        cout << "k_" << i << " = " << k[i] << endl;
    }
    //-------------------------------------------------------------------------------------------------------------------------------
    // TEST
    cout << endl;

    somma_tra_polinomi(encrypted_matrix, encrypted_matrix2, decryptor, parms, context);
    prodotto_polinomio_scalare(encrypted_matrix, 3, decryptor, parms, context);
    somma_polinomio_scalare(encrypted_matrix, 2, k, 0, decryptor, parms, context); //0 sempre fisso con ctr. somma tutti i coefficienti così
    quadrato_dei_coefficienti(encrypted_matrix, decryptor, parms, ev_keys, evaluator, context);
    */
    return 0;
}