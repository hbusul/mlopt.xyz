---
layout: post
title:  "Crossword Puzzle Generator"
date:   2025-02-01 18:41:00 +0100
categories: optimization game
---

While playing a mobile game a couple of months ago, an advertisement caught my
attention. Normally, I would try to skip it immediately but this one was
interesting. Because, you need to solve a crossword puzzle where words consist
of only a subset of letters. Later I found out the game is called
[Wordscapes](https://www.peoplefun.com/games) (not sponsored). Here is
the screenshot:

![Game screenshot](/assets/images/wordscapes.png)

I was immediately interested in representing this game as a mathematical
optimization model. I spent probably an unhealthy amount of my weekend on it
but here it goes.

I'm interested in finding "a" solution rather than "the" solution. Let's start
defining the problem. Given \\(K\\) many words, \\(LW_k\\) denotes the length of
k-th word for \\(\\forall_k \\in [K]\\). For brevity, we will call a row or a
column where you can put a word, a "block". Given \\(J\\) many blocks,
\\(LB_j\\) denotes the length of the j-th block for \\(\forall j \in [J]\\).

We pick the set of words in a way that it only contains words consisting of
selected letters. In addition to that, any word that we cannot fit to any
existing block is excluded from the set of words.

Let's give an example:

![An example configuration](/assets/images/example2.png)

For this configuration we have 4 blocks. \\(LB_1 = 6\\), \\(LB_2 = 5\\),
\\(LB_3 = 4\\) and \\(LB_4 = 3\\). You can arbitrarily number the blocks as long
as it is consistent.

If we were given this problem structure, we would limit our words to words whose
length is 3, 4, 5 or 6. Next information that we need is where 2 blocks
intersect. For example, the first block and the second block intersects at 4th
letter of the first block and at the 1st letter of the second block.

Let I be the set of letter boxes, \\(I = [\\max_{j \\in J}{LB_j}]\\). In this
example, the longest block's length is 6 so \\(I = \\{1, 2, 3, 4, 5, 6\\}\\).
Set \\(common\\) denotes which box intersects with which box and at which
letters. \\(common \\subseteq J \\times J \\times I \\times I\\). For the given
example, \\( (1, 2, 4, 1) \\in common\\) because 1st block and 2nd block
intersect in their 4th and 1st letters.

We know how many boxes there are, their lengths, where they intersect but what
we do not know is that what is the i-th letter of the k-th word. Let \\(L\\) be
be the set of letters that we are allowed to use. For example, for
\\( L = \\{a, c, e, l, p, u\\} \\), the following would be a solution:

![An example configuration](/assets/images/example_sol.png)

We define parameter \\(wlet_{k, i, l}\\) as:

$$
wlet_{k, i, l} = \begin{cases}
    1 & \text{if } \text{i-th letter of the k-th word is l} \\
    0 & \text{otherwise.}
\end{cases}
$$

The main decision in this problem, is deciding which word goes to which block.
\\(w_{j, k} \\in \\{0, 1\\} \\) denotes the decision variable.

$$
w_{j, k} = \begin{cases}
    1 & \text{if } \text{k-th word is assigned to j-th box} \\
    0 & \text{otherwise.}
\end{cases}
$$


If a word does not fit to a block, then we cannot use it there. We can start by
fixing those values:

$$
w_{j, k} = 0 \quad \forall{j, k} : LW_k \neq LB_j
$$

For each block, we need to pick a word:

$$
\sum_{k}{w_{j, k}} = 1 \quad \forall{j \in J}
$$

A word can be used at most once:

$$
\sum_{j}{w_{j, k}} \leq 1 \quad \forall{k \in K}
$$

Now, there comes the more complicated part. For the intersecting blocks, we need
to pick words such that their intersecting letters are the same. Let
\\(j_1, j_2\\) be two intersecting blocks, i.e.
\\(\exists {i_1 \in I, i_2 \in I} : (j_1, j_2, i_1, i_2) \in common\\).

We need to write that, if \\(w_{j_1, k_1} = 1\\) and \\(w_{j_2, k_2} = 1\\),
then

$$
\exists l \in L : wlet_{k_1, i_1, l} = wlet_{k_2, i_2, l} = 1
$$

We can make the statement a bit easier by:

$$
wlet_{k_1, i_1, l} = wlet_{k_2, i_2, l} \quad \forall{l \in L}
$$

One way of writing conditional constraints by using a big-M method. Let's
split the equality into two inequalities.

$$
wlet_{k_1, i_1, l} \leq wlet_{k_2, i_2, l} \quad \forall{l \in L}
$$

$$
wlet_{k_1, i_1, l} \geq wlet_{k_2, i_2, l} \quad \forall{l \in L}
$$

Let's assume we have a condition, \\(cond_{j_1, j_2, k_1, k_2}\\), that is 0
when \\(w_{j_1, k_1} = w_{j_2, k_2} = 1\\); Otherwise
\\(cond_{j_1, j_2, k_1, k_2} \gt 0\\).

$$
wlet_{k_1, i_1, l} \leq wlet_{k_2, i_2, l} + M * cond_{j_1, j_2, k_1, k_2} \quad \forall{l \in L}
$$

$$
wlet_{k_1, i_1, l} + M * cond_{j_1, j_2, k_1, k_2} \geq wlet_{k_2, i_2, l} \quad \forall{l \in L}
$$

So when the \\(cond \gt 0\\), two inequalities are satisfied anyway.
We can write down the condition easily by:

$$
cond_{j_1, j_2, k_1, k_2} = 2 - w_{j_1, k_1} - w_{j_2, k_2}
$$

If we re-write the equations again:

$$
wlet_{k_1, i_1, l} \leq wlet_{k_2, i_2, l} + M * (2 - w_{j_1, k_1} - w_{j_2, k_2}) \quad \forall{l \in L}
$$

$$
wlet_{k_1, i_1, l} + M * (2 - w_{j_1, k_1} - w_{j_2, k_2}) \geq wlet_{k_2, i_2, l} \quad \forall{l \in L}
$$

Instead of writing the equation just for a single \\(j_1, j_2, i_1, i_2\\),
we can write it for all intersections and for all words.


$$
wlet_{k_1, i_1, l} \leq wlet_{k_2, i_2, l} + M * (2 - w_{j_1, k_1} - w_{j_2, k_2}) \quad \forall{l \in L}, \;
\forall{(j_1, j_2, i_1, i_2) \in common}, \; \forall{k_1 \in K}, \; \forall{k_2 \in K}
$$

$$
wlet_{k_1, i_1, l} + M * (2 - w_{j_1, k_1} - w_{j_2, k_2}) \geq wlet_{k_2, i_2, l} \quad \forall{l \in L}, \;
\forall{(j_1, j_2, i_1, i_2) \in common}, \; \forall{k_1 \in K}, \; \forall{k_2 \in K}
$$

This an optimization model where we only seek for a feasible solution, so we do
not need to define an objective.

$$ K: \text{set of words} $$

$$ J: \text{set of blocks} $$

$$ I: \text{set of letter boxes in blocks} $$

$$ L: \text{set of letters allowed} $$

$$ common \subseteq J \times J \times I \times I : \text{set of intersections} $$

$$ LB_j: \text{length of block j} \quad \forall{j \in J} $$

$$ LW_k: \text{length of word k} \quad \forall{k \in K} $$

$$ wlet_{k, i, l}: \text{parameter for representing words' letters}$$

$$ w_{j, k} : \text{variable, 1 if word k is used in block j, 0 otherwise} \quad \forall{k \in K}, \forall{j \in J} $$

$$ w_{j, k} = 0 \quad \forall{j, k} : LW_k \neq LB_j $$

$$ \sum_{k}{w_{j, k}} = 1 \quad \forall{j \in J} $$

$$ \sum_{j}{w_{j, k}} \leq 1 \quad \forall{k \in K} $$

$$
wlet_{k_1, i_1, l} \leq wlet_{k_2, i_2, l} + M * (2 - w_{j_1, k_1} - w_{j_2, k_2}) \quad \forall{l \in L}, \;
\forall{(j_1, j_2, i_1, i_2) \in common}, \; \forall{k_1 \in K}, \; \forall{k_2 \in K}
$$

$$
wlet_{k_1, i_1, l} + M * (2 - w_{j_1, k_1} - w_{j_2, k_2}) \geq wlet_{k_2, i_2, l} \quad \forall{l \in L}, \;
\forall{(j_1, j_2, i_1, i_2) \in common}, \; \forall{k_1 \in K}, \; \forall{k_2 \in K}
$$

Here is the same formulation in [GAMSPy](https://gamspy.readthedocs.io/en/latest/):

``` python
import gamspy as gp

m = gp.Container()

k = gp.Set(
    m, name="k", description="Words that we can choose from"
)
kk = gp.Alias(m, name="kk", alias_with=k)

j = gp.Set(
    m,
    name="j",
    description="set of blocks",
)
jj = gp.Alias(m, "jj", alias_with=j)

i = gp.Set(m, "i", description="set for letter boxes in blocks")
ii = gp.Alias(m, "ii", alias_with=i)


l = gp.Set(m, name="l", description="Letters that are allowed")

common = gp.Set(
    m,
    name="common",
    domain=[j, j, i, i],
    description="Set indicating which char should be common",
)

LB = gp.Parameter(
    m, domain=[j], description="length of the block j"
)

LW = gp.Parameter(
    m, name="LW", domain=[k], description="length of word k"
)

wlet = gp.Parameter(m, name="wlet", domain=[k, i, l])

w = gp.Variable(
    m,
    name="w",
    domain=[j, k],
    type="binary",
    description="which word is assigned to which block",
)

# allow only correct placing
w.fx[j, k].where[LW[k] != LB[j]] = 0

# assign words to every position
fill_blocks = gp.Equation(m, domain=[j])
fill_blocks[j] = gp.Sum(k, w[j, k]) == 1

# use a word atmost once
use_word_atmost_once = gp.Equation(m, domain=[k])
use_word_atmost_once[k] = gp.Sum(j, w[j, k]) <= 1

big_M = 3

# use common letters
intersections_1 = gp.Equation(m, domain=[j, jj, i, ii, l, k, kk])
intersections_1[j, jj, i, ii, l, k, kk].where[common[j, jj, i, ii]] = (
    wlet[k, i, l] + (2 - w[j, k] - w[jj, kk]) * big_M >= wlet[kk, ii, l]
)

intersections_2 = gp.Equation(m, domain=[j, jj, i, ii, l, k, kk])
intersections_2[j, jj, i, ii, l, k, kk].where[common[j, jj, i, ii]] = (
    wlet[k, i, l] <= wlet[kk, ii, l] + (2 - w[j, k] - w[jj, kk]) * big_M
)

model = gp.Model(
    m,
    equations=m.getEquations(),
    problem="mip",
)

```

But no problem is fun without some data that we can play with. We can use the
dictionary coming with most of the linux installations which is at
`/usr/share/dict/american-english`. However filling block related parameters
manually is a cumbersome task. For that, we will use
[streamlit](https://streamlit.io/).

Let's define the imports:

``` python
import uuid
import numpy as np
import streamlit as st
import itertools
from functools import partial

from problem import main, ProblemInput # I'll explain this later
```

Then we need an interface for picking the letters. We want to have something
like this:

![Letter choice part](/assets/images/letters.png)

Doing this in streamlit, is straightforward, you need 1 title and 3 rows filled
with checkboxes.

``` python
row1 = "abcdefgh"
row2 = "ijklmnop"
row3 = "qrstuvwxyz"

letters_selected = {}

st.header("Letters")

cols1 = st.columns(10)
cols2 = st.columns(10)
cols3 = st.columns(10)

for i, r in enumerate(row1):
    with cols1[i]:
        letters_selected[r] = st.checkbox(r, key=f"{r}")

for i, r in enumerate(row2):
    with cols2[i]:
        letters_selected[r] = st.checkbox(r, key=f"{r}")

for i, r in enumerate(row3):
    with cols3[i]:
        letters_selected[r] = st.checkbox(r, key=f"{r}")

letters_selected = {k for k in letters_selected if letters_selected[k]}

st.write("\n")

```

So here `letters_selected` will provide a set of letters that are selected.
Next, we need an interface for selecting the blocks, just like this one:

![Blocks choice part](/assets/images/blocks.png)

Normally, this is easy to achieve with using bunch of checkboxes. However, I did
not like that checkboxes do not give the feeling of selecting boxes for a
crossword puzzle. Instead, we are going to use buttons and kind of create a
toggle button. For a button to remember its state, we need to save if a button
is selected or not in somewhere. For this, we will use
<a href="https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state" target="_blank">
session_state</a> feature of the streamlit.

``` python
row_count = 10
col_count = 10

def toggle_state(row, i):
    st.session_state[(row, i)] = not st.session_state[(row, i)]

st.header("Problem Structure")

for row in range(row_count):
    cols = st.columns(10)
    for i, col in enumerate(cols):
        if (row, i) not in st.session_state:
            st.session_state[(row, i)] = False

        with col:
            color = "primary" if st.session_state[(row, i)] else "secondary"
            st.button(
                key=f"{row}x{i}",
                label=" ",
                on_click=partial(toggle_state, row, i),
                type=color,
                use_container_width=True,
            )
```

This code might look a bit convoluted, but hopefully it will become fairly easy
once we explain it. First, we decide it is going to be a grid that is 10x10.
The idea is that, in our `session_state` we are going to store which of the 100
buttons is selected and which one is not. `toggle_state` function, triggered by
buttons, will simply revert the state once a button is pressed. By default, we
assume no button is pressed so all states are `False`. Later, when we are
creating the buttons, we pick their type depending if they are selected or not.

Selected buttons will have the `primary` (left) type and others will have the
`secondary` (right) type.
![Buttons displaying primary and secondary types](/assets/images/primary_secondary.png)


The next challenge is that getting 100 `True`s and `False`s and converting them
to blocks. Since it is a bit longer code which is included in the codebase, I
will skip it. If you are curious, you can investigate that part yourself.

Then, finally the solution part. We just need a button, when clicked converting
problem into a `ProblemInput` class and then modeling it via GAMSPy and solving
it with a MIP solver.

``` python
from dataclasses import dataclass

coords = tuple[int, int]

@dataclass
class ProblemInput:
    num_blocks: int
    block_lens: list[int]
    min_len: int
    max_len: int
    letters: set[str]
    matches: list[tuple[int, int, int, int]]
    blocks: list[tuple[coords, coords]]
```


Here is the streamlit part that is doing that:

``` python
if st.button("Solve"):
    values = []
    for row in range(row_count):
        temp = []
        for col in range(col_count):
            temp.append(st.session_state[(row, col)])

        values.append(temp)

    values = np.array(values)

    problem = detect_structure(values)
    problem.letters = letters_selected

    result, summary = main(problem)

    new_solution = {}
    for index, word in summary:
        index = int(index)
        block = problem.blocks[index - 1]
        letter_index = 0
        for ih in range(block[0][0], block[1][0] + 1):
            for iw in range(block[0][1], block[1][1] + 1):
                l = word[letter_index]
                letter_index += 1
                new_solution[(ih, iw)] = l

    st.session_state["solution"] = new_solution

    st.write(result)

    for row in range(row_count):
        cols = st.columns(10)
        for i, col in enumerate(cols):
            with col:
                color = "primary" if st.session_state[(row, i)] else "secondary"
                label = " "
                if (row, i) in st.session_state["solution"]:
                    label = st.session_state["solution"][(row, i)]

                st.button(
                    key=f"sol_{row}x{i}",
                    label=label,
                    type=color,
                    use_container_width=True,
                    disabled=True,
                )
```

Let's also explain the last part. From the selected buttons, we find which
blocks exist and where they intersect etc. Then, we feed this input to our
`main` function which expects this certain type of input. So the contract
between streamlit and GAMSPy in this case `ProblemInput` class. Then, we
get the solution and basically print it out in a new section like:

![Solution](/assets/images/solution.png)


I know this has been a bit of a long post, but I hope that it caught your
interest. If you want to play with the application a bit, you can find it
here: [Streamlit App](https://crossword-puzzle.streamlit.app/).
A word of caution, I directly loaded the word list from linux dictionary,
it is not censored and demo license that comes with GAMSPy have size limitations.
However, if you wish to change anything about it or use it with your license
locally, you can find the codebase at
[hbusul/crossword-puzzle](https://github.com/hbusul/crossword-puzzle).

Thanks for reading!

![A cute heart made from blocks](/assets/images/heart.png)
