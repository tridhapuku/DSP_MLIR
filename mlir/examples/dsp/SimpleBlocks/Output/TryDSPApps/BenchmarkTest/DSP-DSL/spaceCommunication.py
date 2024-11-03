def main() {
        var input = "HELLO FROM SPACE";
        # print(input);
        var a = space_modulate(input);
        var noise = sin(a);
        var noisy_signal = a+noise;
        var b = space_demodulate(noisy_signal);
        var e = space_err_correction(b);
        print(e);
}

# ./bin/dsp1 ../mlir/examples/dsp/SimpleBlocks/Output/TryDSPApps/BenchmarkTest/DSP-DSL/spaceCommunication.py -emit=jit 2> input.txt
# ./bin/dsp1 ../mlir/examples/dsp/SimpleBlocks/Output/TryDSPApps/BenchmarkTest/DSP-DSL/spaceCommunication.py -emit=jit 2> output.txt
# diff input.txt ouput.txt