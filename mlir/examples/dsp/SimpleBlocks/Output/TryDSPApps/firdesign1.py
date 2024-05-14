
#fir filter design
#define hid = 1/pi *n sin(wc *n) wc = cut off frequency 
# h[n] = hid[n] * w[n] //w[n] = window like hamming window

def main() {

  #size 10
  var a10 = [ 10,20,30,40,50,60,70,80,90,100];

  var N = 10;

  var wc = 100;
  var hid = sinc(wc, N);
  print(hid);

  # var ones = [1,1,1,1,1,1,1,1,1,1];
  # var pi = 3.14;
  # var hid = div(ones,pi*n)  * sin(wc * n)
  # var ham = hamming(N);
  # var in_mul_ham = a10 * ham;
  # var sliding_out = slidingWindowAvg(in_mul_ham);
  
  # Get fft 
  
  # var c = a10 + a10;
  # var d = c + c;
  # var e = slidingWindowAvg(d);
  # var g1 = downsampling(a10 , 2);
  # var f = upsampling(a10, 8);
  # var g = downsampling(f , 4);
  # var g<8> = downsampling(f , 2);
  # var h<15> = FIRFilter(g,g);
  # var e = delay(c, d);
  # var f = e[0];
  # print(sliding_out);
  # print(d);
  # print(e);

}

