

//Input Len
%0

//FilterLen
%1 

//OutLen
%2 = arith.addi %0, %1
%3 = arith.subi %2, arith.const 1

//run affine loop for output len

affine.for %arg0=0 to %3 //i
    %4 = arith.const 0

    //run affine loop for filter len
    affine.for %arg1=0 to %1 //k
        //check for index i-k
            %5 = %arg0 - %arg1
            affine.if %5 >= 0 and %5 <= %1 - 1
                //load the filter and input values
                %6 = affine.load filter[%arg1]
        //affine-map expression for input-load: affine_map (d0, d1) -> d0 - d1
                %7 = affine-map.apply (%arg0 , %arg1)
                %8 = affine.load input[%7]
                //get the multiply and then sum
                %9 = arith.mulf %6 , %8
                %10 = arith.addf %4, %9
        
    affine.store %10 , out[%arg0]


//code without affine.if
//iterate for each of output 
    //de
    //iterate for each of filter len

for each output o[i]
    temp = 0
    for each filter k
        prod = filter[k] * input[inputlen - k]
        temp = temp + prod
    o[i] = temp

//In affine dialect
affine.for %arg0 = 0 to 10 /*output-len*/{ 
    %1 = arith.constant 0

    affine.for %arg1 = 0 to 10 /*filter-len*/{
        %2 = affine.load filter[%arg1]
        %3 = affine.load input[10 - %arg1]
        %4 = arith.mulf %2 , %3
        %5 = arith.sum %1 , %4
    }

}

//sum of a vector
%1 = arith.const 0
affine.for %arg0 = 0 to 10 /*output-len*/{
    %2 = affine.load input[%arg0]
    %1 = arith.add %1 , %2   
}

affine.store %1, %output

//try affine.if 
#set2 = affine_set<(d0, d1)[]: (d0 - 5 >= 0, d1- 5 >= 0 ) >
affine.for %arg0 = 0 to 10 {
    %N = len(output)
   %4 =  affine.for %arg1 = 0 to 10 {
        affine.if #set2(%arg0 , %arg1 )[%N] {
            %1 = const 5
            %2 = const 3
            %3 = arith.mulf %1 , %2
            affine.yield %3 
        }
    }
   affine.store %4, alloc[%arg0]  
    
}

n-k >= 0 and n-k <= inputLen - 1 ie, n-k + 1 - inputLen <= 0 or, inputLen + k - n- 1 >= 0
Sum = sum + (h[k] * x[n -k] )



//FIRFilter code
//iterate for output
    //start with sum=0
     //iterate for filter len
        //check for input_indx must be within bounds
        //load filter and input[indx]
        //multiply them
        //add this to sum
//update output with sum
#set2 = affine_set<(j) :(10 - j >= 0)>(%j)
#map2 = affine_map<(d1 ) -> (10 - d1) >
affine.for %arg0 = 0 to 10 {
    %1 = arith.const 0
    %sum = affine.for %arg1= 0 to 10 iter_args(%sum_iter = %1) {
        %2 = affine.if #set2(%arg1) {
            %4 = affine.apply #map2(%arg1)
            %5 = affine.load input[%4]
            affine.yield %5
        }
        else
        {
            affine.yield %1
        }

        %3 = affine.load filter[%arg1]
        %6 = arith.mulf %3, %2
        %sum_next = arith.addf %sum_iter , %6
        affine.yield %sum_next
    }
affine.store %sum, alloc[%arg0]
}


#set2 = affine_set<(i,j) :(i - j >= 0, i - j <= N - 1)>(%j)
#map2 = affine_map<(i,j ) -> (i - j) >
affine.for %arg0 = 0 to 10 {
    %1 = arith.const 0
    %sum = affine.for %arg1= 0 to 10 iter_args(%sum_iter = %1) {
        %2 = affine.if #set2(%arg1) {
            %4 = affine.apply #map2(%arg1)
            %5 = affine.load input[%4]
            affine.yield %5
        }
        else
        {
            affine.yield %1
        }

        %3 = affine.load filter[%arg1]
        %6 = arith.mulf %3, %2
        %sum_next = arith.addf %sum_iter , %6
        affine.yield %sum_next
    }
affine.store %sum, alloc[%arg0]
}



//try with direct complete affine ir

module {
  func.func @main() {
    %cst = arith.constant 6.250000e+02 : f64
    %cst_0 = arith.constant 1.000000e+02 : f64
    %cst_1 = arith.constant 9.000000e+01 : f64
    %cst_2 = arith.constant 8.000000e+01 : f64
    %cst_3 = arith.constant 7.000000e+01 : f64
    %cst_4 = arith.constant 6.000000e+01 : f64
    %cst_5 = arith.constant 5.000000e+01 : f64
    %cst_6 = arith.constant 4.000000e+01 : f64
    %cst_7 = arith.constant 3.000000e+01 : f64
    %cst_8 = arith.constant 2.000000e+01 : f64
    %cst_9 = arith.constant 1.000000e+01 : f64
    %cst_10 = arith.constant 0.000000e+00 : f64
    %alloc = memref.alloc() : memref<10xf64>
    %alloc_10 = memref.alloc() : memref<10xf64>
    affine.store %cst_9, %alloc_10[0] : memref<10xf64>
    affine.store %cst_8, %alloc_10[1] : memref<10xf64>
    affine.store %cst_7, %alloc_10[2] : memref<10xf64>
    affine.store %cst_6, %alloc_10[3] : memref<10xf64>
    affine.store %cst_5, %alloc_10[4] : memref<10xf64>
    affine.store %cst_4, %alloc_10[5] : memref<10xf64>
    affine.store %cst_3, %alloc_10[6] : memref<10xf64>
    affine.store %cst_2, %alloc_10[7] : memref<10xf64>
    affine.store %cst_1, %alloc_10[8] : memref<10xf64>
    affine.store %cst_0, %alloc_10[9] : memref<10xf64>
    // #set2 = affine_set<(j) :(10 - j >= 0)>(%j)
    // #map2 = affine_map<(d1 ) -> (10 - d1) >
    affine.for %arg0 = 0 to 10 {
        %1 = arith.const 0
        %sum = affine.for %arg1= 0 to 10 iter_args(%sum_iter = %1) {
            %2 = affine.if #set2(%arg1) {
                %4 = affine.apply #map2(%arg1)
                %5 = affine.load input[%4]
                affine.yield %5
            }
            else
            {
                affine.yield %1
            }

            %3 = affine.load filter[%arg1]
            %6 = arith.mulf %3, %2
            %sum_next = arith.addf %sum_iter , %6
            affine.yield %sum_next
        }
    affine.store %sum, alloc[%arg0]
    }
    dsp.print %alloc : memref<10xf64>
    memref.dealloc %alloc_10 : memref<10xf64>
    memref.dealloc %alloc : memref<10xf64>
    return
  }
}








