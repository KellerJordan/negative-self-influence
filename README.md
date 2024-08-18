# negative-self-influence

## How to run

To demonstrate the phenomenon, run:
```
python trainer_airbench.py
python viz.py
```

This will produce an output like the following.

```
Computing the correct-class margins for each example which was ablated
Example index      margin               self-influence                  p-value
                with    without

Random examples:
0               4.074   4.741           +0.667                          0.0000
1               7.837   7.850           +0.012                          0.7655
2               6.175   6.307           +0.132                          0.0030
3               6.742   6.715           -0.026                          0.5492
4               5.534   5.450           -0.084                          0.0349
5               7.365   7.330           -0.035                          0.4244
6               7.287   7.301           +0.014                          0.8117
7               6.189   6.084           -0.105                          0.0046
8               5.292   5.464           +0.172                          0.0005
9               4.425   4.542           +0.117                          0.0057
10              5.321   5.287           -0.034                          0.4105
11              4.824   4.879           +0.056                          0.2157
12              4.480   4.550           +0.070                          0.0595
13              3.747   4.243           +0.496                          0.0000
14              7.267   7.273           +0.006                          0.8982
15              5.649   5.677           +0.028                          0.5131
16              2.591   2.999           +0.408                          0.0000
17              0.545   1.637           +1.092                          0.0000
18              2.819   3.691           +0.872                          0.0000
19              3.315   4.132           +0.817                          0.0000
Average:                                +0.234

Easy examples:
45114           10.881  10.635          -0.246                          0.0000
47798           9.291   8.829           -0.463                          0.0000
43746           9.138   8.596           -0.542                          0.0000
49106           12.823  12.670          -0.153                          0.0163
47082           11.484  11.145          -0.339                          0.0000
44095           10.477  10.255          -0.222                          0.0005
49524           12.645  12.359          -0.286                          0.0001
41014           13.582  13.380          -0.202                          0.0004
49159           11.024  10.854          -0.170                          0.0104
44279           12.321  12.189          -0.133                          0.0111
45927           12.048  11.903          -0.144                          0.0168
40141           10.347  10.240          -0.107                          0.1160
47731           12.376  12.156          -0.220                          0.0005
46440           8.403   8.280           -0.123                          0.0124
49015           11.456  11.290          -0.167                          0.0005
49690           10.962  10.756          -0.207                          0.0003
46836           11.517  11.296          -0.221                          0.0000
43512           10.185  10.078          -0.107                          0.0460
43189           7.676   7.493           -0.183                          0.0020
49320           10.452  10.248          -0.204                          0.0004
Average:                                -0.222
```

## What's going on here?

The script `trainer_airbench.py` (or `trainer_madry.py`, if you want to use a different training configuration) performs the following steps.

1. Trains 1000 models on the CIFAR-10 training set (as usual).
2. Trains 1000 models on the CIFAR-10 training set, missing 40 specific examples (same number of steps of training, but the model just never sees those 40 examples).
3. Saves the logit outputs of all of the models, on all of the training examples, to disk. (I.e., this would be two tensors of shape `(500, 50000, 10)`)

All of the runs of training are with identical hyperparameters (except the choice of training dataset, of course).

The 40 specific examples are hardcoded. I chose their indices so that:
* The first 20 indices are just `[0, ..., 19]`, which amounts to random examples (since CIFAR-10 is shuffled).
* The last 20 indices are chosen to be easy examples which typically are learned very well (i.e., the trained model typically has high confidence on them).

The point of the experiment is to determine the impact of removing these 40 examples, on their own margins.

The result is as follows:
* For the *random* 20 examples, removing them tends to reduce their confidence (this is quite intuitive).
* For the *easy* 20 examples, *removing* them tends to *increase* their confidence (this is the new phenomenon).

Or equivalently, we can say that *adding* easy examples to the training set *decreases* the trained model's confidence on them. (Which is what's shown in the script output.)

## Implications

* There exists a concrete phenomenon within finite-width neural network learning dynamics which cannot be recapitulated by NTK learning.
* Influence functions cannot even approximate leave-one-out retraining, let alone leave-many-out (because the Hessian is positive-semidefinite).
* We cannot even honestly state that neural network learning minimizes loss!
