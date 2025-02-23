---
author: H. Burak Usul
layout: post
title:  "Combining ML and Optimization"
date:   2025-02-23 22:34:00 +0100
categories: optimization ml
tags: ml machine learning optimization
description: "Why should you care about integrating ML into your decision making?"
image: "/assets/images/combining_ml_and_opt.png"
---

With the rise of large language models (LLMs), there is not a single day passing
without hearing about a new shiny ML technology combined with some existing
technology. To be fair, most of the times they are legit use cases, just not the
shiny one stop solutions that they claim to be. And unfortunately most of the
conversation is only around LLMs but I can understand why:

![An uplifting ChatGPT response](/assets/images/chadgpt.png)

*"yo, Dot, I got you"*

However, today I'll try to give you some solid examples of combining ML and
optimization.

- [Predict, then Optimize](#predict-then-optimize)
- [Verification Problems](#verification-problems)
- [Surrogate Models](#surrogate-models)


## Predict, then Optimize

This might be the most common use. You use Machine Learning to make some
predictions and use those predictions in your optimization problem. Job
definitions are crystal clear, boring but works like a charm.

### Simple Warehouse Problem

Let's borrow the Simple Warehouse problem from Dantzig [[0]](#references).
You have a warehouse where you store some stocks. Initially, you have
\\(stock_{0}\\) many stocks. You can buy or sell stocks. Also storing stocks costs
you \\(storecost\\) per stock. The amount of stocks you can store is limited by
\\(storecap\\) stocks. The price while buying or selling the stocks depends on
the time of the transaction.

Let \\(t \in [T]\\) be set of discrete time-steps where we can buy or sell
some stocks where \\( [T] = \\{1, 2, ..., T\\} \\). Let \\( stock_{t} \\) be
the amount of stocks that we have at the time-step \\(t\\). The amount we buy
is denoted by \\(buy_{t}\\) and similarly the amount that we sell is denoted by
\\(sell_{t}\\). We cannot sell, buy or store negative amounts therefore,
\\( (sell_{t} \geq 0) \land (buy_{t} \geq 0) \land (stock_{t} \geq 0)
\quad \forall{t \in [T]}\\). The amount that we can sell a stock for is
indicated by \\( price_{t} \\). The storage cost is denoted by \\(storecost\\).

Let's write the equation for the stock flow,

$$
stock_{t} = stock_{t - 1} + buy_{t} - sell_{t} \quad \forall{t \in [T]}
$$

where \\(stock_{0}\\) denotes the initial stocks we have. We are limited by the
amount of stocks we can store therefore
\\(stock_{t} \leq storecap \quad \forall{t \in [T]}\\).
Let's define the \\(cost\\):

$$
cost = \sum_{t \in [T]}{price_{t}*(buy_{t} - sell_{t}) + storecost*stock_{t}}
$$

### A problem instance

You need to do quarterly decisions about buying or selling stocks. Initially,
you have 50 stocks, \\(stock_{0} = 50\\). The storage cost is,
\\(storecost\\), 1. And you cannot store more than, \\(storecap\\) 100 stocks.
A minor issue, you do not know the prices!


| Quarter | Price |
|---------|-------|
| Q1      | ??    |
| Q2      | ??    |
| Q3      | ??    |
| Q4      | ??    |



### Knowing the prices

![it's a wazee, it's a wooze, fairy dust](/assets/images/fairy-dust-fugazzi.gif)

Of course, in real life it is quite hard to know how prices for a good will
change. So if you need to make decisions ahead of time, you might need to
predict the prices. Let's say you know the producer uses lumber and steel
during producing a stock.

And you know the historical data:

| Time          | Lumber Price | Steel Price  | Stock Price |
|---------------|--------------|--------------|-------------|
|2023 - January |   460        |  3997        | 9           |
|2023 - April   |   492        |  4027        | 15.5        |
|2023 - July    |   561        |  3711        | 14          |
|2023 - October |   493        |  3586        | 9.5         |
|2024 - January |   574        |  3892        | 18          |
|2024 - April   |   571        |  3331        | 12          |
|2024 - July    |   446        |  3369        | 10          |
|2024 - October |   523        |  3326        | 10          |


You do not have a magical oracle to tell you the prices for 2025, but in the
happy ville, the ruler just introduced some tariffs to your neighbor country
where you import some lumber. So you expect lumber prices to go up and you
make some guesses for 2025.

*Ahem, this is **random** data, please do not use it as a basis for anything!
Any similarity to actual persons, living or dead; or to any situation is purely
coincidental."*

| Time          | Lumber Price | Steel Price  | Stock Price |
|---------------|--------------|--------------|-------------|
|2025 - January |   550        |  3283        | ??          |
|2025 - April   |   592        |  3252        | ??          |
|2025 - July    |   661        |  3575        | ??          |
|2025 - October |   593        |  3586        | ??          |

Data scientist in you wakes up and comes up with the brilliant idea of
estimating the stock price using [Ridge Regression][1] with
[cross validation][2]. Also your inner data scientist is a bit upset about the
number of samples but you hush it.

Let's import our **training data**:
``` python
import numpy as np
from sklearn.linear_model import RidgeCV

lumber = np.array(
    [
        460, 492, 561, 493,
        574, 571, 446, 523,
    ]
)

steel = np.array(
    [
        3997, 4027, 3711, 3586,
        3892, 3331, 3369, 3326,
    ]
)

prices = np.array(
    [
        9, 15.5, 14, 9.5,
        18, 12, 10, 10,
    ]
)
```

We normalize the input data, so that columns have 0-mean and 1-variance:
``` python
avg_steel = np.mean(steel)
std_steel = np.std(steel)

avg_lumber = np.mean(lumber)
std_lumber = np.std(lumber)

lumber = lumber - avg_lumber
lumber = lumber / std_lumber

steel = steel - avg_steel
steel = steel / std_steel
```

Then, we figure out the coefficients:

``` python
X = np.concat((lumber.reshape(8, 1), steel.reshape(8, 1)), axis=1)
y = prices
clf = RidgeCV().fit(X, y)
```

Now, we have an estimator that can predict the stock price from the lumber
and the steel price. Let's put it to work for 2025:

*Again, it is random data!*

``` python
new_lumber = new_lumber - avg_lumber
new_lumber = new_lumber / std_lumber

new_steel = new_steel - avg_steel
new_steel = new_steel / avg_steel

X_new = np.concat((new_lumber.reshape(4, 1), new_steel.reshape(4, 1)), axis=1)
y_new = clf.predict(X_new)

print(y_new)
```

Outputs:
```
[13.44904433 15.05234568 17.8344247  15.22308318]
```

### Optimize by hand

Now, having the price estimates, we can finally start optimizing. But before
writing the code and giving it to a solver, you can give it go and try to come
up with a good solution.

| Quarter | Price Estimate |
|---------|----------------|
| Q1      | 13.5           |
| Q2      | 15             |
| Q3      | 18             |
| Q4      | 15             |

<form>
<fieldset>
<label id="error"></label>
<label id="current-net">Money: 0</label>
<label id="current-quarter">Quarter: Q1</label>
<label id="current-stocks">Current stocks: 50</label>
    <label for="comment-form-name">Amount</label>
    <input type="number" min="0" max="100" name="amount" id="amount" required/>
</fieldset>
<fieldset>
  <button id="buy">Buy</button>
  <button id="sell">Sell</button>
  <button id="restart">Restart</button>
</fieldset>
<ul id="transactions">
</ul>
</form>

### Optimize via a solver

We can easily write down the problem in GAMSPy as follows:

Add the imports:
``` python
import gamspy as gp
import numpy as np
```

Whatever you do within GAMSPy, you will need a container (a GAMS process and
more). Let's create the container and set \\(T\\). Since we have 4 quarters,
it is going to have elements through 1 to 4.
``` python
m = gp.Container()
t = gp.Set(m, name="t", records=range(1, 5))
price = gp.Parameter(
    m,
    name="price",
    domain=[t],
    records=np.array([13.5, 15, 18, 15]),
)

initial_stock = 50
storage_cost = 1
storage_capacity = 100
```

When you provide the inputs to a parameter, if the input is dense you need to
provide it as a NumPy array. Then I define three scalar parameters,
corresponding to \\(stock_{0}\\), \\(storecost\\) and \\(storecap\\).

If a parameter is scalar, I find it easier to use a Python variable instead of a
Parameter but it is just a matter of taste.

Let's define our three variables (\\(stock_{t}\\), \\(buy_{t}\\) and
\\(sell_{t}\\)):

``` python
stock = gp.Variable(m, name="stock", domain=[t], type="Positive")
buy = gp.Variable(m, name="buy", domain=[t], type="Positive")
sell = gp.Variable(m, name="sell", domain=[t], type="Positive")

stock.up[...] = storage_capacity
```

Then we need to define the stock flow and the cost definition:

```python
sb = gp.Equation(m, name="stock_balance", domain=[t])
sb[t] = (
    stock[t]
    == gp.Number(initial_stock).where[gp.Ord(t) == 1]
    + stock[t.lag(1)]
    + buy[t]
    - sell[t]
)


at = gp.Equation(m, name="accounting_cost")
cost = gp.Variable(m)
at[...] = cost == gp.Sum(t, price[t] * (buy[t] - sell[t]) + storage_cost * stock[t])
```

`gp.Number(initial_stock).where[gp.Ord(t) == 1]` ensures for the first quarter,
we start with the initial stocks. `stock[t.lag(1)]` is for \\(stock_{t - 1}\\).

Then we use all the equations, and ask for the minimum cost:


``` python
model = gp.Model(
    m, name="swp", equations=m.getEquations(), objective=cost, sense="min", problem="LP"
)

model.solve()

print(buy.records[["t", "level"]])
print(sell.records[["t", "level"]])
print(cost.toDense())
```

`.records` returns a DataFrame object. Outputs:

```
   t  level
0  1   50.0
1  2    0.0
2  3    0.0
3  4    0.0
   t  level
0  1    0.0
1  2    0.0
2  3  100.0
3  4    0.0
-925.0
```
So if you had 925 at the end of the previous section, congrats you found the
optimal solution!

This was the probably, most common use case. We used ML for prediction, and
based on predictions we made some decisions. People who know
[Smart "Predict, then Optimize"](#references) probably had an expectation of
mentioning the text. But to keep this post shorter, I will write about it as
a continuation of this example.

## Verification Problems

Another use case where you can combine optimization with ML is when you
need to verify a neural network's behaviour using a black-box solver. We cannot
employ neural networks to critical systems without verifying that they are doing
what we expect them to do. Verifying neural nets using solvers is an active
research area. However, I have to admit it is one of the slower methods. But it
is one of the methods that actually proves you the solution it finds the "optimal".

What are some verification problems:
- Robustness
- Fairness
- Monotonicity

### Robustness

You are interested in figuring out whether your neural network changes its
decision or not when minor perturbations introduced to the input. These
minor perturbations, when a neural network is robust, should not change the
decision.

#### Example:

Imagine you are using a neural network to classify street signs, you would not
want it to change its decision when a couple of pixels change their value
slightly.

You can check the docs I wrote at [GAMSPy Docs][3] for checking robustness of
feed-forward networks or convolutional neural networks.


### Fairness

Whenever you have a neural network whose decisions is affecting humans, or
living things if you like, you need to ensure that it makes decisions "fairly".
You need to decide what "fair" is for you but some examples are it should not
be making decisions based upon gender or race. To convice the applicant that
the decision was fair you can present them with the minimal required change in
the application that would make application successful. In literature, also
known as "counter-factual explanations".


#### Example:

Let's say you have a neural net automatically rejecting many loan applications
even before a human looks at them. The last thing you want is that it makes
decisions based on people's gender, ethnicity and so on. You can exclude these
features from your training set. However, the features you excluded might be
easily derivable using other features.


### Monotonicity

If you know that certain input(s) to a neural network should be only positively
or only negatively correlated with a certain output, you can test if that's the
case.


#### Example:

Let's say you trained a neural network to estimate output of a chemical
process, and a certain input chemical should either increase the output
or keep it the same.


## Surrogate Models

In real life, many systems are complicated and hard to write down as clear
algebraic expressions. It is not easy to write down how you see a "spoon".
Likewise, it is not easy to describe what makes a song beautiful. However,
sometimes, you need to make decisions upon such systems. One of the common
use cases is chemical engineering. You can estimate a complex chemical process
using a neural network and then embed this neural network into your optimization
problem, to reduce the cost, increase the yield, etc.

I will not give an example since I want to check with a chemical engineer
first. Hopefully, my next post or the one after that might be about surrogate
models. For now, here is an [OMLT example][4].


## Conclusion

I hope I managed to convince you that there are multitude of reasons why you
might want to combine ML and Optimization. In a single blog post, I cannot
simply cover every use case. Thanks for reading, please let me know if you
have any questions <3.


## References

- Dantzig, G B, Chapter 3.6. In Linear Programming and Extensions. Princeton
University Press, Princeton, New Jersey, 1963.
- Elmachtoub, Adam N., and Paul Grigas. "Smart “predict, then optimize”."
Management Science 68.1 (2022): 9-26.

<script src="/assets/js/whouse.js"></script>

[1]: https://en.wikipedia.org/wiki/Ridge_regression
[2]: https://en.wikipedia.org/wiki/Cross-validation_(statistics)
[3]: https://gamspy.readthedocs.io/en/latest/user/ml/embed_nn.html
[4]: https://github.com/cog-imperial/OMLT/blob/main/docs/notebooks/neuralnet/auto-thermal-reformer-relu.ipynb
