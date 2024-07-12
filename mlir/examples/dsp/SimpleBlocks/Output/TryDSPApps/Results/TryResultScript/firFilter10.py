
def main() {
  # var a = [10,20,30];
  
  #size = 10
#   var input<2,5> = [10,20,30,40,50,60, 70, 80, 90 , 100];
#   var filter<2,5> = [10,20,30,40,50,60, 70, 80, 90 , 100];
  # var input = [10,20,30,40,50,60, 70, 80, 90 , 100];
  # var filter = [10,20,30,40,50,60, 70, 80, 90 , 100];
	# var input = getRangeOfVector(1, 10,1);
  var input = [1,2,3,4];
  var filter = [4,3,2,1];

  var output = FIRFilterResponse(input , filter);
  var len1 = len(output);
  # var output = FIRFilterResSymmOptimized(input , filter);
  print(output);
  print(len1);
}