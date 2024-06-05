def main() {
  var a = [1,2,3];
  var b = [4,5,6];
  var mu = 0.1;
  var iterations = 1000;
  var filterSize = 3;
  var c = lmsFilter(a, b, mu, filterSize, iterations);
  print(c);
}

