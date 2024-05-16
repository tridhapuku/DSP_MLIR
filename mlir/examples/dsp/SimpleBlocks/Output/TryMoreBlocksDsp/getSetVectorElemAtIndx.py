
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


  var a10 = [ 10,20,30];



  var ValatIndx4 = getElemAtIndx(a10 , 2);
  # var ValatIndx0 = getElemAtIndx(a10 , 0);
  # var ValatIndx9 = getElemAtIndx(a10 , 9);
  # var ValatIndx10 = getElemAtIndx(a10 , 10);
  # var NewvalAt4 = [5] * ValatIndx4;
  var b10 = setElemAtIndx(a10 , 2, [22]) ;
  # var b1 = setElemAtIndx(a10 , 2, NewvalAt4);
  # var b2 = setElemAtIndx(a10 , 1, [2] * ValatIndx4);
  # a10 = [ 10,20,3,4,5,60,70,80,90,100]; Not supported
  print(ValatIndx4);
  print(b10);
  # print(b2);
  print(a10);
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

