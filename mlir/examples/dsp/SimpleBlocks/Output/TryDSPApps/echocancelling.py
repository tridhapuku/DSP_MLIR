def main() {
var fs = 8000;
  # var step = 1/8000; 
  # print(step);
  var t = getRangeOfVector(0,100, 0.000125);
  var f_sig = 500;
  var pi = 3.14159265359;
  var getMultiplier = 2 * pi * f_sig;
  var getSinDuration = gain(t, getMultiplier);
  var clean_sig = sin(getSinDuration );

  var echoMultiplier =  getMultiplier;
  var echoSinDuration = gain(t, echoMultiplier);
  var echo_sig = sin(echoSinDuration);

  var noisy_sig = clean_sig + echo_sig;

   # define delay for 3 ms
  var delay1 = 3;
  var echo_signal = delay(noisy_sig, delay1);   

#   print(echo_signal);
  var mu = 0.01;
  var filterSize = 32;
  var y = lmsFilterResponse(echo_signal, clean_sig, mu, filterSize);
  print(y);

}