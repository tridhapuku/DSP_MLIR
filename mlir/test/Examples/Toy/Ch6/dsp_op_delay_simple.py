# RUN: toyc-ch2 %s -emit=mlir 2>&1 | FileCheck %s

# User defined generic function that operates on unknown shaped arguments

# def func1( x , y){
#     var z = x + y;
#     return z;
# }

def main() {
  var a = [10,20,30];
  # var a = [10,20,30,40,50,60, 70, 80, 90 , 100];
  # var a = [ 10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400,410,420,430,440,450,460,470,480,490,500,510,520,530,540,550,560,570,580,590,600,610,620,630,640,650,660,670,680,690,700,710,720,730,740,750,760,770,780,790,800,810,820,830,840,850,860,870,880,890,900,910,920,930,940,950,960,970,980,990,1000 ];
  var b = 2;
  # var d = 4;

  var c = delay(a, b);
  # var e = delay(c, d);
  print(c);
  # print(b);
  # print(g1);
  # print(d);
  # print(g);
  # var a1 = 10;
  # var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
}

# CHECK-LABEL: toy.func @multiply_transpose(
# CHECK-SAME:                               [[VAL_0:%.*]]: tensor<*xf64>, [[VAL_1:%.*]]: tensor<*xf64>) -> tensor<*xf64>
# CHECK:         [[VAL_2:%.*]] = toy.transpose([[VAL_0]] : tensor<*xf64>) to tensor<*xf64>
# CHECK-NEXT:    [[VAL_3:%.*]] = toy.transpose([[VAL_1]] : tensor<*xf64>) to tensor<*xf64>
# CHECK-NEXT:    [[VAL_4:%.*]] = toy.mul [[VAL_2]], [[VAL_3]] :  tensor<*xf64>
# CHECK-NEXT:    toy.return [[VAL_4]] : tensor<*xf64>

# CHECK-LABEL: toy.func @main()
# CHECK-NEXT:    [[VAL_5:%.*]] = toy.constant dense<{{\[\[}}1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
# CHECK-NEXT:    [[VAL_6:%.*]] = toy.reshape([[VAL_5]] : tensor<2x3xf64>) to tensor<2x3xf64>
# CHECK-NEXT:    [[VAL_7:%.*]] = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>
# CHECK-NEXT:    [[VAL_8:%.*]] = toy.reshape([[VAL_7]] : tensor<6xf64>) to tensor<2x3xf64>
# CHECK-NEXT:    [[VAL_9:%.*]] = toy.generic_call @multiply_transpose([[VAL_6]], [[VAL_8]]) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
# CHECK-NEXT:    [[VAL_10:%.*]] = toy.generic_call @multiply_transpose([[VAL_8]], [[VAL_6]]) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
# CHECK-NEXT:    toy.print [[VAL_10]] : tensor<*xf64>
# CHECK-NEXT:    toy.return
