#include <stdio.h>
#include <math.h>

void getRangeOfVector(double* input, int start, int NoOfElements, double Increment) {
    for (int i = 0; i < NoOfElements; i++) {
        input[i] = start + i * Increment;
    }
}

void reverseInput(double* output, double* input, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = input[length - 1 - i];
    }
}

void FIRFilterResponse(double* output, double* input, double* filter, int length) {
    for (int n = 0; n < length; n++) {
        output[n] = 0;
        for (int k = 0; k < length; k++) {
            if (n - k >= 0) {
                output[n] += input[n - k] * filter[k];
            }
        }
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

void squareMagnitude(double* output, double* real, double* imag, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = real[i] * real[i] + imag[i] * imag[i];
    }
}

int main() {
    int length = 10;
    double input[10];
    getRangeOfVector(input, 0, length, 1);

    double reverse_input[10];
    reverseInput(reverse_input, input, length);

    double conv1d[10];
    FIRFilterResponse(conv1d, input, reverse_input, length);

    double fft_real[10];
    double fft_img[10];
    dftReal(fft_real, conv1d, length);
    dftImag(fft_img, conv1d, length);

    double sq[10];
    squareMagnitude(sq, fft_real, fft_img, length);

    for (int i = 0; i < length; i++) {
        printf("%f\n", sq[i]);
    }

    return 0;
}
