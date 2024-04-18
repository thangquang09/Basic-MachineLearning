**TABLE OF CONTENTS**
- [Linear Problem](#linear-problem)
- [The Normal Equation](#the-normal-equation)
- [Computational Complexity](#computational-complexity)

## Linear Problem

A simple regression model of life satisfaction:

$$
life\_satisfaction = \theta_0 + \theta_1 \times GDP_per_capita
$$

This model is just a linear function of the input: `GPD_per_capita`, $\theta_0,\ \theta_1$ are model's parameters.

A linear model make a prediction by simply computing a weighted sum of the input features, plus a contact called the *bias team*. (also called the *intercept term*) as shown in Equation 4.1

$$
\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n \tag{4-1}
$$

In this equation:

- $\hat{y}$ is prediction value
- $n$ is the number of features
- $\theta_i$ are model parameter, including the *bias term* $\theta_0$ and feature weights $\theta_1 ... \theta_n$.
- $x_i$ feature values

Vectorized equation: Equation 4-2

$$
\hat{y} = h_\mathbf{\theta}(\mathbf{x}) = \mathbf{\theta x} \tag{4-2}
$$

In this equation:

- $h_\mathbf{\theta}$ is hypothesis function, using the model parameter $\theta$.
- $\mathbf{theta}$ is the model's parameter vector, containing the *bias term* and *feature weight*.
- $\mathbf{x}$ is the instance's feature vector, containing $x_0, ..., x_n$ with $x_0$ always equal to 1.

Notice that vector in ML is column vector, then the prediction is $\hat{y} = \mathbf{\theta^Tx}$

The MSE of a linear regression hypothesis $h_\mathbf{\theta}$ on a training set $\mathbf{X}$ is calculated using Equation 4-3

$$
MSE(\mathbf{X}, h_\mathbf{\theta}) = \frac{1}{m} \sum_{i=1}^m (\mathbf{\theta^Tx_i} - y_i)^2 \tag{4-3}
$$

To simplify notations, we will just write $MSE(\mathbf{\theta})$ instead of $MSE(\mathbf{X}, h_\mathbf{\theta})$.

## The Normal Equation

$$
\hat{\mathbf{\theta}} = \mathbf{(X^TX)^{-1}X^T y} \tag{4-4}
$$

In this equation:

- $\hat{\mathbf{\theta}}$ is the value of $\mathbf{\theta}$ that minimizes the cost function/
- $\mathbf{y}$ is the vector of target values.

Additionally, $\mathbf{\theta}$ can be computed by below function:

$$
\hat{\mathbf{\theta}} = \mathbf{X^+y}
$$

In this equation:
- $\mathbf{X^+}$ is *pseudoinverse* of $\mathbf{X}$.

The pseudoinverse itself is computed using a standard matrix factorization technique called singular value decomposition (SVD) that can decompose the training set matrix $\mathbf{X}$ into the matrix multiplication of three matrices `!!!! (missing)`
 
This approach is more efficient computing the Normal equation, plus it handles edge cases nicely. The Normal elation may not work if the matrix $\mathbf{X^T X}$ is not invertible, such as $m < n$ or if some feature are redundant, but the pseudoinverse is always defined.

## Computational Complexity

The Normal elation computes the inverse of $\mathbf{X^TX}$ which is an $(n+1) \times (n+1)$ (where $n$ is ^{2.4}$ to $O(n^3)$, depending on the implementation. In other words, if you double the number of features, you multiply the computational time by roughly $2^{2.4} = 5.3$ to $2^3 = 8$

The SVD approach used by Scikit-Learn LinearRegression class is about $O(n^2)$. If you double the number of features, you multiply the computation time by roughly 4.

>[!Note]
>Both the Normal equation and the SVD approach get very slow when the number of features grows large, about 100,000.

Also, once you have trained your linear regression model (using Normal equation or any other algorithm), predictions are very fast.

The computational complexity is linear with regard both the number of instances you want to make the prediction on and the number of features.

In other words, making predictions on twice as many features or instances will take roughly twice as much time.
