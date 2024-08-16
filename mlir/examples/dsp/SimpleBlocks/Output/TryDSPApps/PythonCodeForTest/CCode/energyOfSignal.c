#include <stdio.h>
#include <math.h>

void getRangeOfVector(double* input, int start, int NoOfElements, double Increment) {
    for (int i = 0; i < NoOfElements; i++) {
        input[i] = start + i * Increment;
    }
}

void dftReal(double* real, double* input, int length) {
    for (int k = 0; k < length; k++) {
        real[k] = 0;
        for (int n = 0; n < length; n++) {
            double angle = 2 * M_PI * k * n / length;
            real[k] += input[n] * cos(angle);
        }
    }
}

void dftImag(double* imag, double* input, int length) {
    for (int k = 0; k < length; k++) {
        imag[k] = 0;
        for (int n = 0; n < length; n++) {
            double angle = 2 * M_PI * k * n / length;
            imag[k] -= input[n] * sin(angle);
        }
    }
}

void square(double* output, double* input, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = input[i] * input[i];
    }
}

double sum(double* input, int length) {
    double total = 0;
    for (int i = 0; i < length; i++) {
        total += input[i];
    }
    return total;
}

int len(double* input) {
    return sizeof(input) / sizeof(input[0]);
}

int main() {
    int input_length = 10;
    double input[10];
    getRangeOfVector(input, 0, input_length, 1);

    double fft_real[10];
    double fft_img[10];
    dftReal(fft_real, input, input_length);
    dftImag(fft_img, input, input_length);

    double sq_real[10];
    double sq_img[10];
    square(sq_real, fft_real, input_length);
    square(sq_img, fft_img, input_length);

    double sq_abs[10];
    for (int i = 0; i < input_length; i++) {
        sq_abs[i] = sq_real[i] + sq_img[i];
    }

    double sum1 = sum(sq_abs, input_length);
    int len1 = input_length;
    double res = sum1 / len1;

    printf("%f\n", res);

    return 0;
}
