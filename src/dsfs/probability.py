"""Module containing functions from Chapter 6."""
import math
import random
from collections import Counter

import matplotlib.pyplot as plt


# Uniform distribution
def uniform_pdf(x):
    return 1 if 0 <= x < 1 else 0


def uniform_cdf(x):
    """Returns the probability that uniform random variable is <= x."""
    if x < 0:
        return 0
    elif x < 1:
        return x
    else:
        return 1


# Normal distribution
def normal_pdf(x, mu=0, sigma=1):
    sqrt_two_pi = math.sqrt(2 * math.pi)
    return math.exp(-(x - mu) ** 2 / 2 / sigma ** 2) / (sqrt_two_pi * sigma)


def normal_pdf_example():
    xs = [x / 10.0 for x in range(-50, 50)]
    plt.plot(xs,[normal_pdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')
    plt.plot(xs,[normal_pdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')
    plt.plot(xs,[normal_pdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
    plt.plot(xs,[normal_pdf(x,mu=-1) for x in xs],'-.',label='mu=-1,sigma=1')
    plt.legend()
    plt.title("Various Normal pdfs")
    plt.show()


def normal_cdf(x, mu=0, sigma=1):
    return (1 + math.erf((x-mu) / math.sqrt(2) / sigma)) / 2


def normal_cdf_example():
    xs = [x / 10.0 for x in range(-50, 50)]
    plt.plot(xs, [normal_cdf(x, sigma=1) for x in xs], '-',
             label='mu=0,sigma=1')
    plt.plot(xs, [normal_cdf(x, sigma=2) for x in xs], '--',
             label='mu=0,sigma=2')
    plt.plot(xs, [normal_cdf(x, sigma=0.5) for x in xs], ':',
             label='mu=0,sigma=0.5')
    plt.plot(xs, [normal_cdf(x, mu=-1) for x in xs], '-.',
             label='mu=-1,sigma=1')
    plt.legend(loc=4)  # bottom right
    plt.title("Various Normal cdfs")
    plt.show()


def inverse_normal_cdf(p, mu=0, sigma=1, tolerance=0.00001):
    """Using binary search for finding approximation for normal cdf inversion"""
    # if not standard, rescale standard
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)

    low_z, low_p  = -10.0, 0
    hi_z, hi_p = 10.0, 1
    while hi_z - low_z > tolerance:
        mid_z = (low_z + hi_z) / 2  # midpoint
        mid_p = normal_cdf(mid_z)   # value in midpoint
        if mid_p < p:
            # midpoint too low, go above
            low_z, low_p = mid_z, mid_p
        elif mid_p > p:
            # midpoint too high, go below
            hi_z, hi_p = mid_z, mid_p
        else:
            break
    return mid_z


# Central limit theorem
def bernoulli_trial(p):
    return 1 if random.random() < p else 0


def binomial(n, p):
    return sum(bernoulli_trial(p) for _ in range(n))


def make_hist(p, n, num_points):
    data = [binomial(n, p) for _ in range(num_points)]

    # use a bar chart to show actual samples
    histogram = Counter(data)
    plt.bar(histogram.keys(),
            [v / num_points for v in histogram.values()],
            0.8,
            color='0.75')
    mu = p * n
    sigma = math.sqrt(n * p * (1 - p))
    # use a line chart to show the normal approximation
    xs = range(min(data), max(data) + 1)
    ys = [normal_cdf(i + 0.5, mu, sigma) - normal_cdf(i - 0.5, mu, sigma)
          for i in xs]
    plt.plot(xs, ys)
    plt.title("Binomial Distribution vs. Normal Approximation")
    plt.show()