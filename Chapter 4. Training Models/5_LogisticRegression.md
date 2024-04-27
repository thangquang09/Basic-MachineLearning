# Logistic Regression

Logistic Regression also called *logit regression* is commonly used to estimate the probability that an instance belongs to a particular class. If the estimated probability is greater than a given threshold $\to$ predict 1, otherwise $\to$ predict 0.

## Estimated Probabilities

Like Linear Regression, Logistic Regression computes a weighted sum of the input features (plus a bias term), but instead of outputting the result directly like linear regression, it outputs the *logistic* of this result.

$$
\hat{p} = h_\mathbf{\theta}(\mathbf{x}) = \sigma(\mathbf{\theta^T x}) \tag{4-13}
$$

$\sigma$ is sigmoid function (also called logistic function) that outputs a number between 0 and 1:

$$
\sigma(t) = \frac{1}{1 + \exp(-t)}
$$

![sigmoid function](images/logistic_function.png)

Relationship between $\hat{p}$ and $\hat{y}$:

$$
\hat{y} = \begin{cases}
    0 & \text{if } & \hat{p} < 0.5 \\
    1 & \text{if } & \hat{p} \geq 0.5
\end{cases}
$$

Notice: $\sigma(t) < 0.5$ when $t < 0$ and $\sigma(t) \geq 0.5$ when $t \geq 0$, so a logistic regression model using the default threshold of $50\%$ probability predicts $1$ if $\mathbf{\theta^T x} \geq 0$ and $0$ if $\mathbf{\theta^T x} < 0$.

The score $t$ is often called the *logit*. The name comes from the fact that the logit function, defined as $\text{logit}(p) = \text{logit}(p / (1-p))$.

