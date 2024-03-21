
//code for the expected fir --

//InputLen 
%0 

//FilterLen 
%1

//OutputLen
%2 = arith.addi %0,%1
%3 = arith.subi %2, arith.const 1

for %arg0 =0 to %3 //i
    %4 = arith.const 0

    for %arg1= 0 to %1 //k
        %5 = %arg0 - %arg1    //i -k 
        arith.if %5 >= 0 and %5 < %0
            %6 = affine.load filter[%arg1]
            %7 = affine.load input[%]
          
            %4 = %4 





