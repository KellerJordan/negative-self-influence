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

Random examples: tensor([ 0.6666,  0.0123,  0.1321, -0.0265, -0.0839, -0.0352,  0.0142, -0.1050,
         0.1719,  0.1173, -0.0340,  0.0557,  0.0698,  0.4957,  0.0059,  0.0279,
         0.4084,  1.0917,  0.8721,  0.8173])
Easy examples: tensor([-0.2464, -0.4627, -0.5420, -0.1528, -0.3387, -0.2222, -0.2860, -0.2019,
        -0.1699, -0.1326, -0.1442, -0.1068, -0.2205, -0.1230, -0.1667, -0.2066,
        -0.2205, -0.1073, -0.1828, -0.2036])

Random mean: tensor(0.2337)
Easy mean: tensor(-0.2219)
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
