
# var c = filter(b,a,x);
# filter b,a should be of same length and equal to x and also, a[0] should be 1
# if not equal in length, append 0

def main() {
    
    #size 10
  # var a10 = [ 10,20,30,40,50,60,70,80,90,100];
  var a10 = [ 10,20,30,40,50];
  var b = [4,2,3,4];
  var a = [1,2,3,4];
  #size=100
  # var a100 = [ 10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400,410,420,430,440,450,460,470,480,490,500,510,520,530,540,550,560,570,580,590,600,610,620,630,640,650,660,670,680,690,700,710,720,730,740,750,760,770,780,790,800,810,820,830,840,850,860,870,880,890,900,910,920,930,940,950,960,970,980,990,1000 ];  

  #size 10
  # var N = 4;
  # var d = square(a10);
  # var c = sum(a10);
  var e = fft1dreal(a10);
  var f = fft1dimg(a10);
  # filter b,a should be of same length and also, a[0] should be 1
  # var c = filter(a10,a10,a10);
  # var e = delay(c, d);
  # var f = e[0];
  # print(c);
  # print(d);
  print(e);
  print(f);
  # print(N);
  # print(e);

}

