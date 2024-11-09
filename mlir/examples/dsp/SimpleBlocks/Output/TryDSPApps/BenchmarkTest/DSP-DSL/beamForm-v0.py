def main() {

    var num_antennas = 4;
    var num_samples = 100;
    var frequency = 5;
    var time = getRangeOfVector(0, 100, 1);
    var weights = [1,2,3,4];

    var signal = beam_form(num_antennas, frequency, time, weights);
    print(signal);
}