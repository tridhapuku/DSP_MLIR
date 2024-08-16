#include <stdio.h>
#include <math.h>

void getRangeOfVector(double* vector, double start, int length, double increment) {
    for (int i = 0; i < length; i++) {
        vector[i] = start + i * increment;
    }
}

void gain(double* output, double* input, double multiplier, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = input[i] * multiplier;
    }
}

void sine(double* output, double* input, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = sin(input[i]);
    }
}

void add(double* output, double* input1, double* input2, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = input1[i] + input2[i];
    }
}

void lmsFilterResponse(double* output, double* noisy_sig, double* clean_sig, double mu, int filterSize, int length) {
    double w[32] = {0};
    for (int n = 0; n < length; n++) {
        double y = 0;
        for (int i = 0; i < filterSize; i++) {
            if (n - i >= 0) {
                y += w[i] * noisy_sig[n - i];
            }
        }
        double e = clean_sig[n] - y;
        for (int i = 0; i < filterSize; i++) {
            if (n - i >= 0) {
                w[i] += mu * e * noisy_sig[n - i];
            }
        }
        output[n] = y;
    }
}

int main() {
    int length = 100;
    double fs = 8000;
    double t[100];
    getRangeOfVector(t, 0, length, 0.000125);

    double f_sig = 500;
    double pi = 3.14159265359;
    double getMultiplier = 2 * pi * f_sig;

    double getSinDuration[100];
    gain(getSinDuration, t, getMultiplier, length);

    double clean_sig[100];
    sine(clean_sig, getSinDuration, length);

    double f_noise = 3000;
    double getNoiseMultiplier = 2 * pi * f_noise;

    double getNoiseSinDuration[100];
    gain(getNoiseSinDuration, t, getNoiseMultiplier, length);

    double noise[100];
    sine(noise, getNoiseSinDuration, length);

    double noise1[100];
    gain(noise1, noise, 0.5, length);

    double noisy_sig[100];
    add(noisy_sig, clean_sig, noise1, length);

    double mu = 0.01;
    int filterSize = 32;
    double y[100];
    lmsFilterResponse(y, noisy_sig, clean_sig, mu, filterSize, length);

    double sol[100];
    gain(sol, y, 10, length);

    for (int i = 0; i < length; i++) {
        printf("%f\n", sol[i]);
    }

    return 0;
}
