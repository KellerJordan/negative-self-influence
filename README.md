# negative-self-influence

Demonstrating a phenomenon in deep learning

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

