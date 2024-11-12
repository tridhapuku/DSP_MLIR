% Constants
INPUT_LENGTH = 8;

% Generate binary input data
data = generate_input(INPUT_LENGTH);

% Perform QAM modulation
symbols = qam_modulate(data);

% Perform QAM demodulation
demodulated_bits = qam_demodulate(symbols);

% Display results
// disp('Original Data:');
// disp(data);

// disp('Modulated Symbols:');
// disp(symbols);

disp('Demodulated Bits:');
disp(demodulated_bits);

%% Function Definitions

% Function to generate random binary input data
function data = generate_input(length)
    data = randi([0 1], 1, length);
end

% Function to perform QAM modulation
function symbols = qam_modulate(data)
    numSymbols = length(data) / 2;
    symbols = complex(zeros(1, numSymbols));
    for i = 1:2:length(data)
        bit1 = data(i);
        bit2 = data(i + 1);

        if bit1 == 0 && bit2 == 0
            symbols((i + 1) / 2) = -1 - 1j;
        elseif bit1 == 0 && bit2 == 1
            symbols((i + 1) / 2) = -1 + 1j;
        elseif bit1 == 1 && bit2 == 0
            symbols((i + 1) / 2) = 1 - 1j;
        else
            symbols((i + 1) / 2) = 1 + 1j;
        end
    end
end

% Function to perform QAM demodulation
function bits = qam_demodulate(symbols)
    numBits = 2 * length(symbols);
    bits = zeros(1, numBits);
    for i = 1:length(symbols)
        symbol = symbols(i);

        if symbol == -1 - 1j
            bits(2 * i - 1) = 0;
            bits(2 * i) = 0;
        elseif symbol == -1 + 1j
            bits(2 * i - 1) = 0;
            bits(2 * i) = 1;
        elseif symbol == 1 - 1j
            bits(2 * i - 1) = 1;
            bits(2 * i) = 0;
        else
            bits(2 * i - 1) = 1;
            bits(2 * i) = 1;
        end
    end
end
