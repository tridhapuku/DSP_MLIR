def main() {
        # var input = [1,0,1,1,0,1,0,0];
	var input = getRangeOfVector(0, 100, 0.000125);
        # print(c);
        var pi = 3.14159265359;
        var f_sig = 500;
        var getMultiplier = 2 * pi * f_sig;
        # print(getMultiplier);
        var getSinDuration = gain(input, getMultiplier);
        # print(getSinDuration);
        var clean_sig = sin(getSinDuration );
        var binary_sig = thresholdUp(clean_sig, 0.4,0);
        # print(binary_sig);
        var modulate_symbol_real = qam_modulate_real(binary_sig);
        # print(modulate_symbol_real);
        var modulate_symbol_imagine = qam_modulate_imagine(binary_sig);
        # print(modulate_symbol_imagine);
        var decode_data = qam_demodulate(modulate_symbol_real, modulate_symbol_imagine);
        # print(decode_data);
        var final2 = getElemAtIndx(modulate_symbol_imagine , 2);
        var final1 = getElemAtIndx(decode_data , 2);
        print(final1);
        print(final2);
}

