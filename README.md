# negative-self-influence

This repository demonstrates a newly observed phenomenon in deep learning:

*Adding an easy example to the training set tends to cause the retrained model to become **less** confident on it.*

## How to run

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
0               4.074   4.741           +0.667                          0.0000
2               6.175   6.307           +0.132                          0.0030
7               6.189   6.084           -0.105                          0.0046
8               5.292   5.464           +0.172                          0.0005
9               4.425   4.542           +0.117                          0.0057
13              3.747   4.243           +0.496                          0.0000
16              2.591   2.999           +0.408                          0.0000
17              0.545   1.637           +1.092                          0.0000
18              2.819   3.691           +0.872                          0.0000
19              3.315   4.132           +0.817                          0.0000
Average:                                +0.467

Easy examples:
45114           10.881  10.635          -0.246                          0.0000
47798           9.291   8.829           -0.463                          0.0000
43746           9.138   8.596           -0.542                          0.0000
47082           11.484  11.145          -0.339                          0.0000
44095           10.477  10.255          -0.222                          0.0005
49524           12.645  12.359          -0.286                          0.0001
41014           13.582  13.380          -0.202                          0.0004
47731           12.376  12.156          -0.220                          0.0005
49015           11.456  11.290          -0.167                          0.0005
49690           10.962  10.756          -0.207                          0.0003
46836           11.517  11.296          -0.221                          0.0000
43189           7.676   7.493           -0.183                          0.0020
49320           10.452  10.248          -0.204                          0.0004
Average:                                -0.269
```

## What's going on here?

The script `trainer_airbench.py` (or `trainer_madry.py`, if you want to use a different training configuration) performs the following steps.

1. Trains 1000 models on the CIFAR-10 training set (as usual).
2. Trains 1000 models on the CIFAR-10 training set, missing 40 specific examples (same number of steps of training, but the model just never sees those 40 examples).
3. Saves the logit outputs of all of the models, on all of the training examples, to disk. (I.e., two tensors of shape `(500, 50000, 10)`)

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

## Implications

* There exists a concrete phenomenon within finite-width neural network learning dynamics which cannot be recapitulated by NTK learning. (Related work [Allen-Zhu & Li (2020)](https://arxiv.org/abs/2012.09816))
* Influence functions cannot even approximate leave-one-out retraining, let alone leave-many-out (because the Hessian is positive-semidefinite). (Related work [Bae et al. (2022)](https://arxiv.org/abs/2209.05364))
* Neural network learning does not minimize loss. (This is totally new!)

## Appendix

What happens if we try with the Madry trainer instead (which is quite different, within the space of CIFAR-10 trainings)?

Here's the output I got from doing so with 1000 total runs.

```
Showing examples whose estimated self-influence is statistically-significantly different from zero (p < 0.01):

Example index      margin               self-influence                  p-value
                with    without

Random examples:
0               5.265   6.253           +0.988                          0.0000
1               13.491  13.228          -0.263                          0.0082
2               8.977   9.326           +0.349                          0.0005
13              4.495   5.601           +1.106                          0.0000
16              4.519   4.822           +0.303                          0.0005
17              -0.161  1.784           +1.945                          0.0000
18              3.675   5.075           +1.400                          0.0000
19              4.257   5.565           +1.308                          0.0000
Average:                                +0.892

Easy examples:
47798           21.260  20.867          -0.394                          0.0071
47731           20.137  19.710          -0.428                          0.0018
49015           21.050  20.716          -0.334                          0.0068
Average:                                -0.385
```

Looks like the same thing, so the phenomenon is not just specific to `airbench` trainings.

