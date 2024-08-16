#include <stdio.h>
#include <math.h>
#include <stdlib.h>

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

void threshold(double* output, double* input, double thresh, int length) {
    for (int i = 0; i < length; i++) {
        if (input[i] >= thresh || input[i] <= -thresh) {
            output[i] = input[i];
        } else {
            output[i] = 0;
        }
    }
}

void quantization(double* output, double* input, int nlevels, double max, double min, int length) {
    double step = (max - min) / nlevels;
    for (int i = 0; i < length; i++) {
        output[i] = round((input[i] - min) / step) * step + min;
    }
}

int* runLenEncoding(double* input, int length, int* rleLength) {
    int* rle = (int*)malloc(length * sizeof(int));
    int index = 0;
    for (int i = 1; i < length; i++) {
        if (input[i] != input[i - 1]) {
            rle[index++] = input[i - 1];
            rle[index++] = 1;
        } else {
            rle[index - 1]++;
        }
    }
    *rleLength = index;
    return rle;
}

double getElemAtIndx(int* rle, int indx) {
    return rle[indx];
}

int main() {
    int input_length = 50000;
    double input[50000];
    getRangeOfVector(input, 0, input_length, 1);

    int nlevels = 16;
    double min = 0;
    double max = 8;

    double threshold_val = 4;

    double fft10real[50000];
    double fft10img[50000];
    
    dftReal(fft10real, fft10img, input, input_length);
    dftImag(fft10real, fft10img, input, input_length);

    double GetThresholdReal[50000];
    double GetThresholdImg[50000];
    threshold(GetThresholdReal, fft10real, threshold_val, input_length);
    threshold(GetThresholdImg, fft10img, threshold_val, input_length);

    double QuantOutReal[50000];
    double QuantOutImg[50000];
    quantization(QuantOutReal, GetThresholdReal, nlevels, max, min, input_length);
    quantization(QuantOutImg, GetThresholdImg, nlevels, max, min, input_length);

    int rleLengthReal, rleLengthImg;
    int* rLEOutReal = runLenEncoding(QuantOutReal, input_length, &rleLengthReal);
    int* rLEOutImg = runLenEncoding(QuantOutImg, input_length, &rleLengthImg);

    double final1 = getElemAtIndx(rLEOutReal, 6);
    double final2 = getElemAtIndx(rLEOutImg, 7);
    printf("%f\n", final1);
    printf("%f\n", final2);

    free(rLEOutReal);
    free(rLEOutImg);

    return 0;
}
