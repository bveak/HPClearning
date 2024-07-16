== n = 50000, dim = 1 << 14 ==

GPU time(ms): 578.807068
GPU time(ms): 13.136608
GPU time(ms): 13.140128
GPU time(ms): 13.130080
GPU time(ms): 13.141696
GPU time(ms): 13.123872
GPU time(ms): 13.132128
GPU time(ms): 13.130784
GPU time(ms): 13.139584
GPU time(ms): 13.143296
GPU average time(ms): 13.136799
CPU time(ms): 428.452000
Accepted!!

暂时没搞明白为啥有时候第一次会慢，暂时用去掉最大最小值的方法算平均了

== n = 50000 << 14, dim = 1 ==

GPU time(ms): 13.945536
GPU time(ms): 15.458560
GPU time(ms): 13.668800
GPU time(ms): 13.642336
GPU time(ms): 13.668352
GPU time(ms): 13.662208
GPU time(ms): 13.661280
GPU time(ms): 13.649248
GPU time(ms): 13.702912
GPU time(ms): 13.649824
GPU average time(ms): 13.701019
CPU time(ms): 560.464000
task1: main.cu:64: int main(): Assertion `std::fabs((ansput[i] - output[i]) / ansput[i]) < 1e-3' failed.

这个 fail 是因为太多数字加在一起精度不够

== n = 50000 << 9, dim = 32 ==

GPU time(ms): 13.861856
GPU time(ms): 13.650048
GPU time(ms): 13.633664
GPU time(ms): 13.650048
GPU time(ms): 13.660032
GPU time(ms): 13.646400
GPU time(ms): 13.629440
GPU time(ms): 13.647328
GPU time(ms): 13.653312
GPU time(ms): 13.652992
GPU average time(ms): 13.649228
CPU time(ms): 405.506000
task1: main.cu:64: int main(): Assertion `std::fabs((ansput[i] - output[i]) / ansput[i]) < 1e-3' failed.

== n = 50000 << 12, dim = 4 ==

GPU time(ms): 13.965920
GPU time(ms): 13.630112
GPU time(ms): 13.665632
GPU time(ms): 13.662240
GPU time(ms): 13.626752
GPU time(ms): 13.649280
GPU time(ms): 13.686880
GPU time(ms): 13.645792
GPU time(ms): 13.641920
GPU time(ms): 13.605312
GPU average time(ms): 13.651077
CPU time(ms): 629.312000
task1: main.cu:64: int main(): Assertion `std::fabs((ansput[i] - output[i]) / ansput[i]) < 1e-3' failed.

感觉好像怎么跑都没差了。。。