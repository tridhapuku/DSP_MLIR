import numpy as np
import matplotlib.pyplot as plt

# Define constants
N = 1000       # Length of signal
M = 32         # Number of filter coefficients
MU = 0.01      # Step size (learning rate)
fs = 8000      # Sampling rate (Hz)
t = np.arange(0, N) / fs  # Time vector

# Generate a clean sine wave (signal frequency 500 Hz)
f_signal = 200  # Signal frequency (Hz)
clean_signal = np.sin(2 * np.pi * f_signal * t)

# Generate an echo (delayed and scaled version of the clean signal)
delay = int(0.01 * fs)  # 10 ms delay
echo_scale = 2       # Echo scaling factor
echo_signal = np.zeros_like(clean_signal)
echo_signal[delay:] = echo_scale * clean_signal[:-delay]

# Generate the primary signal (clean signal + echo)
primary_signal = clean_signal + echo_signal

# LMS filter function
def lms_filter(x, d):
    w = np.zeros(M)
    y = np.zeros(len(x))
    e = np.zeros(len(x))

    for n in range(len(x)):
        x_vec = np.concatenate((x[n:n-M:-1], np.zeros(M-len(x[n:n-M:-1]))))
        y[n] = np.dot(w, x_vec)
        e[n] = d[n] - y[n]
        w += MU * e[n] * x_vec

    return y, e, w

# Apply LMS filter for echo cancellation
output_signal, error_signal, filter_weights = lms_filter(primary_signal, clean_signal)

# Plot the signals
plt.figure(figsize=(15, 10))

plt.subplot(4, 1, 1)
plt.plot(clean_signal)
plt.title('Clean Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

plt.subplot(4, 1, 2)
plt.plot(primary_signal)
plt.title('Primary Signal (with echo)')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

plt.subplot(4, 1, 3)
plt.plot(echo_signal)
plt.title('Echo Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

plt.subplot(4, 1, 4)
plt.plot(output_signal)
plt.title('Output Signal (echo cancelled)')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()
