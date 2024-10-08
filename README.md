# negative-self-influence

This repository demonstrates a newly observed phenomenon in deep learning:

*Adding an easy example to the training set tends to cause the retrained model to become **less** confident on it.*

## Implications

* There exists a concrete phenomenon within finite-width neural network learning dynamics which cannot be recapitulated by NTK learning. (Related work [Allen-Zhu & Li (2020)](https://arxiv.org/abs/2012.09816))
* Influence functions cannot even approximate leave-one-out retraining, let alone leave-many-out (because the Hessian is positive-semidefinite). (Related work [Bae et al. (2022)](https://arxiv.org/abs/2209.05364))
* Neural network learning does not minimize loss. (This is new)


## How to run the experiment

To demonstrate the phenomenon, run:
```
python trainer_airbench.py
python viz.py
```

This will produce an output like the following.

```
Showing examples whose estimated self-influence is statistically-significantly different from zero (p < 0.01):

Example index      margin               self-influence                  p-value
                with    without

Random examples:
0               4.741   4.074           +0.667                          0.0000
2               6.307   6.175           +0.132                          0.0030
7               6.084   6.189           -0.105                          0.0046
8               5.464   5.292           +0.172                          0.0005
9               4.542   4.425           +0.117                          0.0057
13              4.243   3.747           +0.496                          0.0000
16              2.999   2.591           +0.408                          0.0000
17              1.637   0.545           +1.092                          0.0000
18              3.691   2.819           +0.872                          0.0000
19              4.132   3.315           +0.817                          0.0000
Average:                                +0.467

Easy examples:
45114           10.635  10.881          -0.246                          0.0000
47798           8.829   9.291           -0.463                          0.0000
43746           8.596   9.138           -0.542                          0.0000
47082           11.145  11.484          -0.339                          0.0000
44095           10.255  10.477          -0.222                          0.0005
49524           12.359  12.645          -0.286                          0.0001
41014           13.380  13.582          -0.202                          0.0004
47731           12.156  12.376          -0.220                          0.0005
49015           11.290  11.456          -0.167                          0.0005
49690           10.756  10.962          -0.207                          0.0003
46836           11.296  11.517          -0.221                          0.0000
43189           7.493   7.676           -0.183                          0.0020
49320           10.248  10.452          -0.204                          0.0004
Average:                                -0.269
```

## What's going on here?

The script `trainer_airbench.py` (or `trainer_madry.py`, if you want to use a different training configuration) performs the following steps.

1. Trains 1000 models on the CIFAR-10 training set (as usual).
2. Trains 1000 models on the CIFAR-10 training set, missing 40 specific examples (same number of steps of training, but the model just never sees those 40 examples).
3. Saves the logit outputs of all of the models, on all of the training examples, to disk. (I.e., two tensors of shape `(1000, 50000, 10)`)

All of the runs of training are with otherwise identical hyperparameters.

The 40 specific examples are hardcoded. I chose their indices so that:
* The first 20 indices are just `[0, ..., 19]`, which amounts to random examples (since CIFAR-10 is shuffled).
* The last 20 indices are chosen to be easy examples which are known to have negative self-influence for the `airbench` trainer.

The point of the experiment is to determine the impact of removing these 40 examples, on their own margins.

The result is as follows:
* For the *random* 20 examples, removing them tends to reduce their confidence (this is quite intuitive).
* For the *easy* 20 examples, *removing* them tends to *increase* their confidence (this is the new phenomenon).

Or equivalently, we can say that *adding* easy examples to the training set *decreases* the trained model's confidence on them. (Which is what's shown in the script output.)

## Why wasn't this phenomenon observed before?

Presumably because doing 1,000 trainings to get statistical significance on the differences is a pain in the butt.

## Appendix

What happens if we try with the Madry trainer instead (which is quite different, within the space of CIFAR-10 trainings)?

Here's the output I got from doing so with 4000 total runs.

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

Looks like the same thing, so the phenomenon is not just specific to `airbench` trainings.

And here's another run with the Madry trainer, with cutout augmentation turned off.
```
Computing the correct-class margins for each of the 40 examples which were ablated

Showing examples whose estimated self-influence is statistically-significantly different from zero (p < 0.01):

Example index      margin               self-influence                  p-value
                with    without

Random examples:
0               5.260   6.472           +1.212                          0.0000
2               9.116   9.438           +0.322                          0.0000
5               8.660   8.801           +0.141                          0.0026
8               8.276   8.452           +0.176                          0.0032
13              4.225   5.368           +1.143                          0.0000
16              3.756   4.388           +0.632                          0.0000
17              -0.177  2.149           +2.326                          0.0000
18              3.965   5.260           +1.295                          0.0000
19              4.256   5.499           +1.243                          0.0000
Average:                                +0.943

Easy examples:
47798           21.559  21.239          -0.320                          0.0002
47082           19.311  18.983          -0.328                          0.0004
44095           18.063  17.718          -0.345                          0.0000
41014           26.358  26.110          -0.248                          0.0006
47731           20.699  20.465          -0.234                          0.0037
Average:                                -0.295
```

Again the same thing.

