#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define INPUT_CHAR_LENGTH 16 // number of characters
#define INPUT_LENGTH INPUT_CHAR_LENGTH*8 // number of bits

void charToBinary(char c, char *binaryStr) {
    for (int i = 7; i >= 0; i--) {
        binaryStr[7 - i] = ((c >> i) & 1) ? '1' : '0';
    }
    binaryStr[8] = '\0'; // Null-terminate the binary string
}

// 1. Signal Conditioning (convert to binary)
void condition_signal(const char* data, char* binary) {
    binary[0] = '\0';  // Ensure binary string starts empty
    for (int i = 0; i < strlen(data); i++) {
        char bin[9];
        charToBinary(data[i], bin);
        strcat(binary, bin);
    }
}

// 2. Modulation (simple BPSK modulation)
void modulate(const char* binary_data, int* modulated_signal) {
    for (int i = 0; i < strlen(binary_data); i++) {
        modulated_signal[i] = (binary_data[i] == '1') ? 1 : -1;
    }
}

// 3. Transmission and Reception (add noise to simulate)
void transmit_and_receive(const int* signal, double* received_signal, int length, double noise_level) {
    for (int i = 0; i < length; i++) {
        double noise = sin(signal[i]);
        received_signal[i] = signal[i] + noise;
    }
}

// 4. Demodulation
void demodulate(const double* signal, char* demodulated_data, int length) {
    for (int i = 0; i < length; i++) {
        demodulated_data[i] = (signal[i] > 0) ? '1' : '0';
    }
    demodulated_data[length] = '\0'; // Null-terminate the demodulated data
}

// 5. Error Correction (simple parity check)
void error_correction(const char* data, char* corrected) {
    int length = strlen(data);
    int corrected_index = 0;
    for (int i = 0; i < length; i += 8) {
        int count = 0;
        for (int j = 0; j < 8; j++) {
            if (data[i + j] == '1') count++;
        }
        if (count % 2 == 0) {
            strncpy(&corrected[corrected_index], &data[i], 8);
        } else {
            corrected[corrected_index] = '0';
            strncpy(&corrected[corrected_index + 1], &data[i + 1], 7);
        }
        corrected_index += 8;
    }
    corrected[corrected_index] = '\0';
}

// 6. Data Decoding
void decode_data(const char* binary, char* decoded) {
    int length = strlen(binary);
    int decoded_index = 0;
    for (int i = 0; i < length; i += 8) {
        char byte[9];
        strncpy(byte, &binary[i], 8);
        byte[8] = '\0';
        decoded[decoded_index++] = (char)strtol(byte, NULL, 2);
    }
    decoded[decoded_index] = '\0'; // Null-terminate the decoded data
}

int main() {
    const char* space_data = "HELLO FROM SPACE";
    char* binary;
    binary = (char*)malloc(INPUT_LENGTH * sizeof(char));

    for(int i=0; i<INPUT_LENGTH; i++) {
        binary[i] = '0';
    }

    int* modulated_signal;
    modulated_signal = (int*)malloc(INPUT_LENGTH * sizeof(int));

    double* received_signal;
    received_signal = (double*)malloc(INPUT_LENGTH * sizeof(double));

    char* demodulated_data;
    demodulated_data = (char*)malloc(INPUT_LENGTH * sizeof(char));

    char* corrected_signal;
    corrected_signal = (char*)malloc(INPUT_LENGTH * sizeof(char));

    char* decoded_data;
    decoded_data = (char*)malloc(INPUT_CHAR_LENGTH * sizeof(char));

    // 1. Signal Conditioning
    condition_signal(space_data, binary);

    // 2. Modulation
    modulate(binary, modulated_signal);

    // 3. Transmission and Reception
    transmit_and_receive(modulated_signal, received_signal, strlen(binary), 0.1);

    // 4. Demodulation
    demodulate(received_signal, demodulated_data, strlen(binary));

    // 5. Error Correction
    error_correction(demodulated_data, corrected_signal);

    // 6. Data Decoding
    decode_data(corrected_signal, decoded_data);

    printf("Original data: %s\n", space_data);
    printf("Decoded data: %s\n", decoded_data);

    return 0;
}
