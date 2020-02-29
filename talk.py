import streamlit as st

st.title("A Careful Walk Through Probability with Python")

"""Eric J. Ma, PyCon 2020, Pittsburgh, PA"""

"""
## What is a probability distribution?

Let's explore it as a Python object.

I've got a definition:

> A probability distribution is a Python object
> that assigns credibility points to values
> on the number line.

You've seen probability distributions before.
For simplicity, we'll start with the Gaussian,
also known as the Normal distribution.

## SciPy Probability Distributions

This is a great place to start.
Let's look at the Gaussian distribution
and what the SciPy stats library provides
as methods that we can call on.
"""

with st.echo():
    from scipy.stats import norm
    my_dist = norm(loc=0, scale=1)

"""
Here, we've set the "state" of the probability distribution
to have a location/mean of `0`,
and a scale/standard deviation of `1`.
Once we've set the state of the probability distribution,
we can interact with it.

### Credibility Points

The mapping from x-value to crediblity points
is provided by the probability density function,
also known as the "PDF".
"""

with st.echo():
    x_value = st.slider(
        "x",
        min_value=-3.,
        max_value=3.,
        step=0.1
    )
    credibility_points = my_dist.pdf(x_value)
    st.write(f"The credibility points associated with {x_value} is {credibility_points:.3f}")


"""
We can visualize this across all x-values.
"""
import matplotlib.pyplot as plt
from utils import plot_distribution
import numpy as np
import pandas as pd


fig, ax = plt.subplots()
ax = plot_distribution(my_dist, ax)
ax.set_title("A Gaussian PDF")
st.pyplot(fig)

"""
The usual notation one will use is that

- the data are $x$
- the credibility points are $P(x)$

## Drawing Numbers

Given a distribution, we can also draw
a sequence of numbers from it.
The notation here would be:

$X = [x_1, x_2, x_3, ...]$

Drawing one number would give me:
"""

with st.echo():
    xs = my_dist.rvs(1)
    xs

"""
Drawing 10 numbers would give me:
"""

with st.echo():
    xs = my_dist.rvs(10)
    xs

"""
In SciPy distribution land,
drawing numbers is done by calling on the
`.rvs(n)` class method,
passing in `n` the number of draws you wish to take.

We call this __sampling__ from a probability distribution.

## i.i.d sampling

A very important concept here is the idea that
the value of the number we drew from the probability distribution
didn't depend on the value of the previous one.

The **total credibility points** associated with the draws
is the **product** of the credibility points. It is not the sum.
(_Accept this as a definition for now._)


We can calculate the total credibility points,
or __total probability__
associated with 10 draws from the Gaussian.
Mathematically, it is:

$x_1 \\times x_2 \\times ... \\times x_n = \\prod_{i=1}^{n} x_i$
"""

with st.echo():
    xs = my_dist.rvs(10)
    probabilities = my_dist.pdf(xs)
    total_probability = np.product(probabilities)
    total_probability

"""
That's a very small number!
If we drew more, soon we'd encounter underflow issues.
Thus, summing the log of probabilities is a more common thing to do.
"""

with st.echo():
    log_probs = my_dist.logpdf(xs)
    total_logprobs = np.sum(log_probs)
    total_logprobs

"""
We've seen now that we can do things with
"""
