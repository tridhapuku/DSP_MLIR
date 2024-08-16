#include <stdio.h>
#include <math.h>

#define PI 3.14159265359

void getRangeOfVector(double* input, int start, int NoOfElements, double Increment) {
    for (int i = 0; i < NoOfElements; i++) {
        input[i] = start + i * Increment;
    }
}

void gain(double* output, double* input, double multiplier, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = input[i] * multiplier;
    }
}

void elementWiseAdd(double* output, double* input1, double* input2, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = input1[i] + input2[i];
    }
}

void elementWiseMultiply(double* output, double* input1, double* input2, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = input1[i] * input2[i];
    }
}

void lowPassFIRFilter(double* lpf, double wc, int N) {
    int mid = (N - 1) / 2;
    for (int n = 0; n < N; n++) {
        if (n == mid) {
            lpf[n] = wc / PI;
        } else {
            lpf[n] = (wc / PI) * sin(wc * (n - mid)) / (wc * (n - mid));
        }
    }
}

void hammingWindow(double* hamming, int N) {
    for (int n = 0; n < N; n++) {
        hamming[n] = 0.54 - 0.46 * cos(2 * PI * n / (N - 1));
    }
}

void FIRFilterResponse(double* output, double* input, double* filter, int input_length, int filter_length) {
    int i, j;
    for (i = 0; i < input_length; i++) {
        output[i] = 0;
        for (j = 0; j < filter_length; j++) {
            if (i - j >= 0) {
                output[i] += input[i - j] * filter[j];
            }
        }
    }
}

int main() {
    int fs = 8000;
    int input_length = 30;
    double input[30];
    getRangeOfVector(input, 0, input_length, 0.000125);

    double f_sig = 500;
    double getMultiplier = 2 * PI * f_sig;

    double getSinDuration[30];
    gain(getSinDuration, input, getMultiplier, input_length);
    
    double clean_sig[30];
    for (int i = 0; i < input_length; i++) {
        clean_sig[i] = sin(getSinDuration[i]);
    }

    double f_noise = 3000;
    double getNoiseSinDuration[30];
    gain(getNoiseSinDuration, input, 2 * PI * f_noise, input_length);
    
    double noise[30];
    for (int i = 0; i < input_length; i++) {
        noise[i] = sin(getNoiseSinDuration[i]);
    }

    double noisy_sig[30];
    double scaled_noise[30];
    gain(scaled_noise, noise, 0.5, input_length);
    elementWiseAdd(noisy_sig, clean_sig, scaled_noise, input_length);

    double fc = 1000;
    double wc = 2 * PI * fc / fs;
    int N = 101;
    double lpf[101];
    lowPassFIRFilter(lpf, wc, N);

    double hamming[101];
    hammingWindow(hamming, N);

    double lpf_w[101];
    elementWiseMultiply(lpf_w, lpf, hamming, N);

    double FIRfilterResponse[30];
    FIRFilterResponse(FIRfilterResponse, noisy_sig, lpf_w, input_length, N);

    for (int i = 0; i < input_length; i++) {
        printf("%f\n", FIRfilterResponse[i]);
    }

    return 0;
}
