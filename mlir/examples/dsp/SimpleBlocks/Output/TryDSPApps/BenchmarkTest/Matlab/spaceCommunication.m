% Parameters
INPUT_CHAR_LENGTH = 16; 
INPUT_LENGTH = INPUT_CHAR_LENGTH * 8; 

% Original data
space_data = 'HELLO FROM SPACE';

% 1. Signal Conditioning (Convert characters to binary)
binary_data = '';
for i = 1:length(space_data)
    binary_data = strcat(binary_data, dec2bin(space_data(i), 8));
end

% 2. Modulation (Simple BPSK Modulation)
modulated_signal = zeros(1, length(binary_data));
for i = 1:length(binary_data)
    if binary_data(i) == '1'
        modulated_signal(i) = 1;
    else
        modulated_signal(i) = -1;
    end
end

% 3. Transmission and Reception (Add noise to simulate)
noise_level = 0.1;
received_signal = modulated_signal + noise_level * randn(1, length(modulated_signal));

% 4. Demodulation
demodulated_data = '';
for i = 1:length(received_signal)
    if received_signal(i) > 0
        demodulated_data = strcat(demodulated_data, '1');
    else
        demodulated_data = strcat(demodulated_data, '0');
    end
end

% 5. Error Correction (Simple Parity Check)
corrected_signal = '';
for i = 1:8:length(demodulated_data)
    byte = demodulated_data(i:i+7);
    ones_count = sum(byte == '1');
    if mod(ones_count, 2) == 0
        corrected_signal = strcat(corrected_signal, byte);
    else
        corrected_signal = strcat(corrected_signal, '0', byte(2:end));
    end
end

% 6. Data Decoding
decoded_data = '';
for i = 1:8:length(corrected_signal)
    byte = corrected_signal(i:i+7);
    decoded_data = strcat(decoded_data, char(bin2dec(byte)));
end

% Display Results
fprintf('Original data: %s\n', space_data);
fprintf('Decoded data: %s\n', decoded_data);
