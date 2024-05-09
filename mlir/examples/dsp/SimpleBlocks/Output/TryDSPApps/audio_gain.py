
def audio_gain(a10 , ham) {
    
    var out = a10 * ham;
    print(out);
}


def main() {

  #size 10
  var a10 = [ 10,20,30,40,50,60,70,80,90,100];

  var N = 10;
  var Fs = 10000 ;
  var F1 = 1000 ;
  var F2 = 5000;
  var G1 = 20;
  var G2 = 10;

  var k1 = F1 * N / Fs; #formula for k1
  print(k1);
  # var 
  # var g1 =
  #Get fft : %real , %img = 
  # Multiply real & img with gain -g1 , g2 for indx k1 & k2 
  # 
  #compute ifft 
  #ifft of ifa_real , ifa_complex for real part
  #ifft of ifb_real , ifb_complex for img part 
  #now: final_out_real = ifa_real - ifb_complex

  # audio_gain(a10, ham);
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
  print(in_mul_ham);
  # print(d);
  # print(e);

}

