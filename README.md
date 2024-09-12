# Formula Discovery using SA/RLSA

This is a Pytorch implementation for the paper "**A two-stage symbolic regression method for discovering mathematical formulas**\".

We propose a method that uses prior knowledge to generate reasonable mathematical terms from given variables and then uses simulated annealing (SA) or reinforcement learning enhanced SA (RLSA) to discover mathematical relationships between those terms.


## Environment
- python 3.7
- numpy 1.9.1
- sympy 1.8
- pytorch 1.8.1
- sklearn 0.23.2

## Datasets
As pointed out in the paper, we use SA or RLSA to select an optimal subset of terms from a set of terms to construct a mathematical equation, here the terms are obtained by leveraging prior knowledge in elementary mathematics. Just keep it simple and to the point, we provide the terms and their corresponding value directly in this program. 

In the first dataset TriangleAreaFormula/datasets, the terms are built from **a**, ***b***, **c,** **α**, **β**, and **γ**, which are the three sides and three angles of a triangle.

In the second dataset TrigonometricFormula/datasets, the terms are built from two angles, i.e., **A** and **B**.


## Implementation
The proposed method is successfully applied to two problems in elementary mathematics: discovering the triangle area formulas and the trigonometric formulas. Our method can find 295 triangle area formulas, which is more than the 251 formulas listed in the literature, and it can find 752 trigonometric formulas.

#### **TriangleAreaFormula**

We propose two methods, i.e., SA and RLSA, to discover triangle area formulas. You can run the demo program according to the following steps.

1. perform SA by running 

   ```
   python TriangleAreaFormula/SA.py
   ```

2. We provided a trained model, so perform RLSA directly by running 

   ```
   python TriangleAreaFormula/RLSA.py
   ```

   Once you have created your training and validation datasets, run

   ```
   python TriangleAreaFormula/train.py
   ```

Their results are stored separately in

```
./result/SA_result.npy
./result/RLSA_result.npy
```

In the directory “.**/result** “, the results of one run and all the triangle area formulas that we found are provided.

#### TrigonometricFormula

Only SA is used to find the trigonometric formulas. You can run the demo program according to the following steps. 

```
python TrigonometricFormula/main.py 
```

Its results are stored in

```
./result/result.npy
```

In the directory “**./result** “, the results of one run and all the trigonometric formulas that we found are provided.