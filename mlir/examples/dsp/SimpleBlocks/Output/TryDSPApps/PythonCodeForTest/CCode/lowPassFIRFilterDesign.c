#include <stdio.h>
#include <math.h>
#include <stdlib.h>

void hamming(double* window, int N) {
    for (int i = 0; i < N; i++) {
        window[i] = 0.54 - 0.46 * cos(2 * M_PI * i / (N - 1));
    }
}

void lowPassFIRFilter(double* filter, double wc, int N) {
    int mid = (N - 1) / 2;
    for (int n = 0; n < N; n++) {
        if (n == mid) {
            filter[n] = wc / M_PI;
        } else {
            filter[n] = sin(wc * (n - mid)) / (M_PI * (n - mid));
        }
    }
}

void elementWiseMultiplication(double* output, double* array1, double* array2, int N) {
    for (int i = 0; i < N; i++) {
        output[i] = array1[i] * array2[i];
    }
}

double getElemAtIndx(double* array, int index) {
    return array[index];
}

int main() {
    int N = 51;
    double pi = 3.14159265359;
    double fc1 = 500;
    double Fs = 8000;
    double wc1 = 2 * pi * fc1 / Fs;

    double lpf[51];
    lowPassFIRFilter(lpf, wc1, N);

    double hamming_window[51];
    hamming(hamming_window, N);

    double lpf_w[51];
    elementWiseMultiplication(lpf_w, lpf, hamming_window, N);

    double final1 = getElemAtIndx(lpf_w, 6);

    printf("%f\n", final1);

    return 0;
}
