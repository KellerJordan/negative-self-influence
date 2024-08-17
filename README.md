# negative-self-influence

## How to run

To demonstrate the phenomenon, run:
```
python trainer_airbench.py
python viz.py
```

This will produce an output like the following.

```
The impact of *including* each of the 40 examples on its own correct-class margin was as follows...

Random examples: tensor([ 0.7093, -0.0059,  0.1041, -0.0505, -0.0219,  0.0603, -0.0410,  0.0474,
         0.0161,  0.0102,  0.1198,  0.0163,  0.0930,  0.5851, -0.0943,  0.0376,
         0.4222,  1.1793,  0.9131,  0.8261])
Easy examples: tensor([ 0.1002,  0.0241, -0.1136,  0.0268,  0.0267, -0.0046, -0.0301, -0.0469,
         0.0692, -0.0396, -0.2784, -0.2612, -0.4000, -0.3311, -0.1263, -0.3525,
        -0.3298, -0.1946, -0.3078, -0.2419, -0.1945, -0.1482, -0.3998, -0.0911,
        -0.2081, -0.1319, -0.0461, -0.1823, -0.1905, -0.0830])

Random mean: tensor(0.2463)
Easy mean: tensor(-0.1496)
```

## What's going on here?

The script `trainer_airbench.py` (or `trainer_madry.py`, if you want to use a different training configuration) performs the following steps.

1. Trains 500 models on the CIFAR-10 training set (as usual).
2. Trains 500 models on the CIFAR-10 training set, missing 40 specific examples (same number of steps of training, but the model just never sees those 40 examples).
3. Saves the logit outputs of all of the models, on all of the training examples, to disk. (I.e., this would be two tensors of shape `(500, 50000, 10)`)

All of the runs of training are with identical hyperparameters (except the choice of training dataset, of course).

The 40 specific examples are hardcoded. I chose their indices so that:
* The first 20 indices are just `[0, ..., 19]`, which amounts to random examples (since CIFAR-10 is shuffled).
* The last 20 indices are chosen to be easy examples which typically are learned very well (the trained model has high confidence on them).

The point of the experiment is to determine the impact of removing those 40 examples, on the margins of those 40 examples.

The result is as follows:
* On the first 20 examples (random), removing them reduces their confidence (this is quite intuitive).
* On the second 20 examples (easy), *removing* them *increases* their confidence (this is the new phenomenon).

Or equivalently, we can say that *adding* easy examples to the training set *decreases* the trained model's confidence on them. (Which is what's shown in the script output.)
