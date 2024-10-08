# Results

`python viz.py 1` (madry trainer)
```
Showing examples whose estimated self-influence is statistically-significantly different from zero (p < 0.01):

Example index      margin               self-influence                  p-value
                with    without

Random examples:
0               6.310   5.088           +1.222                          0.0000
2               9.396   9.088           +0.308                          0.0000
8               8.463   8.235           +0.228                          0.0000
11              7.629   7.475           +0.153                          0.0016
13              5.706   4.521           +1.185                          0.0000
16              4.918   4.543           +0.375                          0.0000
17              1.805   -0.078          +1.882                          0.0000
18              5.068   3.743           +1.325                          0.0000
19              5.511   4.359           +1.152                          0.0000
Average:                                +0.870

Easy examples:
47798           20.914  21.178          -0.263                          0.0003
43746           21.084  21.401          -0.317                          0.0000
47731           19.915  20.149          -0.234                          0.0005
Average:                                -0.271
```

`python viz.py 10` (airbench94)
```
Showing examples whose estimated self-influence is statistically-significantly different from zero (p < 0.01):

Example index      margin               self-influence                  p-value
                with    without

Random examples:
0               4.596   3.909           +0.687                          0.0000
2               6.126   5.942           +0.185                          0.0000
13              4.176   3.716           +0.461                          0.0000
16              2.950   2.638           +0.312                          0.0000
17              1.631   0.548           +1.083                          0.0000
18              3.669   2.831           +0.838                          0.0000
19              4.087   3.314           +0.773                          0.0000
Average:                                +0.620

Easy examples:
45114           10.720  11.042          -0.321                          0.0000
47798           8.912   9.316           -0.404                          0.0000
43746           8.669   9.056           -0.386                          0.0000
49106           12.486  12.654          -0.168                          0.0068
47082           10.981  11.253          -0.272                          0.0002
44095           10.246  10.515          -0.269                          0.0000
49524           12.288  12.500          -0.212                          0.0024
41014           13.235  13.415          -0.179                          0.0019
40141           9.991   10.254          -0.263                          0.0001
47731           11.851  12.036          -0.185                          0.0037
46440           8.301   8.546           -0.246                          0.0000
49015           11.158  11.308          -0.150                          0.0015
49690           10.741  10.914          -0.173                          0.0038
46836           11.308  11.515          -0.207                          0.0000
43512           9.940   10.090          -0.150                          0.0048
43189           7.445   7.647           -0.202                          0.0003
Average:                                -0.237
```

`python viz.py 11` (clean_airbench94)
```
Showing examples whose estimated self-influence is statistically-significantly different from zero (p < 0.01):

Example index      margin               self-influence                  p-value
                with    without

Random examples:
0               4.604   3.925           +0.679                          0.0000
2               6.201   5.947           +0.254                          0.0000
11              4.865   4.713           +0.152                          0.0012
13              4.138   3.689           +0.449                          0.0000
16              2.894   2.600           +0.294                          0.0000
17              1.622   0.562           +1.060                          0.0000
18              3.657   2.747           +0.910                          0.0000
19              3.993   3.301           +0.692                          0.0000
Average:                                +0.561

Easy examples:
45114           10.987  11.235          -0.248                          0.0001
47798           9.168   9.597           -0.429                          0.0000
43746           8.950   9.436           -0.486                          0.0000
49106           12.590  12.779          -0.189                          0.0039
49524           12.367  12.566          -0.199                          0.0063
41014           13.407  13.558          -0.151                          0.0093
49159           11.208  11.406          -0.198                          0.0058
44279           12.252  12.418          -0.166                          0.0031
45927           11.814  11.980          -0.166                          0.0081
40141           10.192  10.423          -0.230                          0.0011
46440           8.422   8.696           -0.274                          0.0000
49015           11.204  11.394          -0.191                          0.0001
46836           11.452  11.690          -0.238                          0.0000
43189           7.646   7.815           -0.169                          0.0041
49320           10.253  10.410          -0.157                          0.0081
Average:                                -0.233
```


`python viz.py 12` (remove bias scaler)
```
Showing examples whose estimated self-influence is statistically-significantly different from zero (p < 0.01):

Example index      margin               self-influence                  p-value
                with    without

Random examples:
0               4.189   3.819           +0.370                          0.0000
2               5.173   4.934           +0.239                          0.0000
8               4.516   4.308           +0.208                          0.0000
9               3.959   3.767           +0.191                          0.0000
13              3.937   3.535           +0.402                          0.0000
15              4.702   4.594           +0.108                          0.0027
16              2.602   2.304           +0.298                          0.0000
17              0.999   0.204           +0.795                          0.0000
18              3.079   2.388           +0.691                          0.0000
19              3.829   3.249           +0.580                          0.0000
Average:                                +0.388

Easy examples:
45114           10.737  10.952          -0.214                          0.0000
47798           10.180  10.408          -0.228                          0.0000
43746           9.498   9.713           -0.215                          0.0000
44095           9.451   9.616           -0.165                          0.0017
Average:                                -0.206
```

`python viz.py 13` (switch to RenormSGD)
```
Showing examples whose estimated self-influence is statistically-significantly different from zero (p < 0.01):

Example index      margin               self-influence                  p-value
                with    without

Random examples:
0               4.225   3.848           +0.378                          0.0000
2               5.447   5.208           +0.239                          0.0000
4               6.612   6.703           -0.091                          0.0075
8               4.493   4.320           +0.173                          0.0003
13              4.094   3.621           +0.473                          0.0000
16              2.652   2.377           +0.274                          0.0000
17              1.028   0.158           +0.870                          0.0000
18              3.528   2.878           +0.649                          0.0000
19              4.005   3.601           +0.404                          0.0000
Average:                                +0.374

Easy examples:
47798           10.375  10.687          -0.312                          0.0000
43746           9.729   10.001          -0.272                          0.0000
41014           13.480  13.607          -0.128                          0.0075
49690           11.626  11.789          -0.162                          0.0007
Average:                                -0.218
```

`python viz.py 14` (epochs=20, bs=500, momentum=0.9, no bias scaler, RenormSGD)
```
Showing examples whose estimated self-influence is statistically-significantly different from zero (p < 0.01):

Example index      margin               self-influence                  p-value
                with    without

Random examples:
0               4.535   3.594           +0.941                          0.0000
2               6.825   6.635           +0.190                          0.0001
8               4.796   4.598           +0.198                          0.0000
9               4.215   3.890           +0.325                          0.0000
13              4.987   4.289           +0.698                          0.0000
15              5.838   5.686           +0.151                          0.0003
16              2.809   2.154           +0.655                          0.0000
17              2.240   -0.379          +2.619                          0.0000
18              5.956   5.345           +0.611                          0.0000
19              4.315   3.480           +0.836                          0.0000
Average:                                +0.722

Easy examples:
45114           10.734  11.132          -0.398                          0.0000
47798           8.748   9.189           -0.441                          0.0000
43746           8.258   8.746           -0.488                          0.0000
49106           12.481  12.690          -0.209                          0.0007
47082           11.630  11.808          -0.179                          0.0041
44095           10.165  10.448          -0.283                          0.0000
49524           13.807  14.109          -0.302                          0.0000
41014           14.864  15.097          -0.232                          0.0000
49159           9.845   10.068          -0.223                          0.0005
44279           12.144  12.311          -0.167                          0.0004
45927           11.369  11.605          -0.236                          0.0000
40141           9.170   9.418           -0.247                          0.0000
47731           12.628  12.935          -0.307                          0.0000
46440           7.534   7.738           -0.205                          0.0000
49015           12.184  12.374          -0.189                          0.0001
49690           10.833  11.080          -0.246                          0.0000
46836           11.487  11.689          -0.201                          0.0000
43512           9.511   9.670           -0.159                          0.0005
49320           9.931   10.120          -0.189                          0.0001
Average:                                -0.258
```

`python viz.py 15` (rm the BatchNorms; 1.5% lower accuracy) (n=2000 each)
```
Showing examples whose estimated self-influence is statistically-significantly different from zero (p < 0.01):

Example index      margin               self-influence                  p-value
                with    without

Random examples:
0               8.135   7.055           +1.081                          0.0000
2               10.124  9.923           +0.201                          0.0001
8               11.267  11.051          +0.216                          0.0030
11              10.645  10.496          +0.149                          0.0085
13              8.758   7.881           +0.877                          0.0000
15              9.712   9.515           +0.198                          0.0000
16              6.774   6.563           +0.211                          0.0000
17              2.140   -0.090          +2.230                          0.0000
18              5.525   4.274           +1.251                          0.0000
19              7.510   5.936           +1.575                          0.0000
Average:                                +0.799

Easy examples:
49106           21.681  21.460          +0.221                          0.0055
47731           16.613  16.470          +0.143                          0.0056
46836           30.410  30.062          +0.348                          0.0000
Average:                                +0.237
```

`python viz.py 17` (LayerNorm)
```
Showing examples whose estimated self-influence is statistically-significantly different from zero (p < 0.01):

Example index      margin               self-influence                  p-value
                with    without

Random examples:
0               4.462   3.795           +0.667                          0.0000
2               4.752   4.705           +0.047                          0.0000
9               4.981   4.934           +0.047                          0.0005
13              4.400   4.153           +0.248                          0.0000
15              4.346   4.238           +0.108                          0.0000
16              3.501   3.038           +0.463                          0.0000
17              2.072   0.264           +1.808                          0.0000
18              3.862   3.218           +0.644                          0.0000
19              4.829   4.428           +0.401                          0.0000
Average:                                +0.493

Easy examples:
47731           5.954   5.970           -0.017                          0.0026
43512           6.352   6.367           -0.015                          0.0069
Average:                                -0.016
```

`python viz.py 18` (LayerNorm with better params)
```
Showing examples whose estimated self-influence is statistically-significantly different from zero (p < 0.01):

Example index      margin               self-influence                  p-value
                with    without

Random examples:
0               4.844   3.827           +1.016                          0.0000
13              5.844   5.311           +0.533                          0.0000
15              5.618   5.474           +0.144                          0.0003
16              3.535   2.892           +0.643                          0.0000
17              2.255   0.442           +1.813                          0.0000
18              4.215   3.336           +0.878                          0.0000
19              5.404   4.840           +0.563                          0.0000
Average:                                +0.799

Easy examples:
46440           11.045  10.995          +0.051                          0.0089
Average:                                +0.051
```

