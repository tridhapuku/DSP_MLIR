
#Code Usage for setElemAtIndx & getElemAtIndx: 
# var b10 = setElemAtIndx(a10 , 2, [22]) ;
    # MLIR-Affine
    #     affine.store %cst, %alloc_4[0] : memref<1xf64>
    #     %0 = affine.load %alloc_4[0] : memref<1xf64>
    #     affine.store %0, %alloc_6[2] : memref<3xf64>

# var ValatIndx4 = getElemAtIndx(a10 , 2); ;
    # MLIR-Affine
    #     affine.store %cst, %alloc_4[0] : memref<1xf64>
    #     %0 = affine.load %alloc_4[0] : memref<1xf64>
    #     affine.store %0, %alloc_6[2] : memref<3xf64>



def main() {

  #size 10
  # var g1 = 6;
  var a10 = [ 10,20,30,40,50,60,70,80,90,100];

  # var N = 10;
  # var Fs = 10000 ; #Sampling freq
  # var F1 = 1000 ;
  # var F2 = 5000;
  var G1 = 4;
  # var G2 = 2;

  # k1 & k2 are indexes
  # var k1 = (F1 * N) / Fs; #formula for k1
  var k1 = [1];
  var k2 = [4];

  # var k2 = F2 * N / Fs;
  var fft10real = fft1dreal(a10);
  var fft10img = fft1dimg(a10);
  var ValAtK1real = getElemAtIndx(fft10real , k1); #k1
  var ValAtK1img = getElemAtIndx(fft10img , k1);

  # var ValAtK2real = getElemAtIndx(a10 , k1);

  #Apply gain at index k1 & k2
  var modifiedValAtk1real = gain(ValAtK1real , G1);
  var modifiedValAtk1img = gain(ValAtK1img , G1);
  # var modifiedValAtk2 = gain(ValatIndxK2 , G2);

  #set the values at index 
  var b1 = setElemAtIndx(fft10real , k1 , modifiedValAtk1real);
  var b2 = setElemAtIndx(fft10img , k1 , modifiedValAtk1img);
  # var b2 = setElemAtIndx(fft10 , k2 , modifiedValAtk2);

  #Do ifft
  # print(fft10);
  # print(fft10img);
  var res1 = ifft1d(fft10real , fft10img);
  print(b1);
  print(b2);
  print(res1);

  # print(b1);
  # print(b2);
  # print(fft10);
  # var ValatIndx0 = getElemAtIndx(a10 , 0);
  # var ValatIndx9 = getElemAtIndx(a10 , 9);
  # var ValatIndx10 = getElemAtIndx(a10 , 10);
  # var NewvalAt4 = [5] * ValatIndx4;
  # var b10 = setElemAtIndx(a10 , 2, [22]) ;
  # # var b1 = setElemAtIndx(a10 , 2, NewvalAt4);
  # # var b2 = setElemAtIndx(a10 , 1, [2] * ValatIndx4);
  # # a10 = [ 10,20,3,4,5,60,70,80,90,100]; Not supported
  # # print(ValatIndx4);
  # print(b10);
  # # print(b2);
  # print(a10);
  # print(ValatIndx0);
  # print(ValatIndx9);
  # print(ValatIndx10);
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
  # print(in_mul_ham);
  # print(d);
  # print(e);

}

