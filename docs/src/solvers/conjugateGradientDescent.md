# Conjugate Gradient Descent

```@docs
conjugateGradientDescent
```
The conjugate gradient descent extends the classical [Gradient Descent](@ref)
as follows: for some $x_0\in\mathcal M$ the algorithm computes

$ x_{k+1} = \exp_{x_k} \alpha_k\delta_k $

where $\alpha_k$ stems from some line search and the direction $\delta_k$ is
updated with

$ \delta_{k+1} = -\xi_{k+1} + \beta_k\delta_k,\quad \delta_0 = -\xi_0 $
where $\xi_k = \nabla f(x_k)$ is the gradient of the cost function $f$ to be minimized.

The coefficient $\beta_k$ is computed based on $\xi_k,\delta_k\in T_{x_{k}}\mathcal M$,
and $\xi_{k+1}\in T_{x_{k+1}}\mathcal M$. Adapting the Euclidean rules hence requires
a parallel transport $P_{x_k\to x_{k+1}}$. The following rules are available.

```@docs
steepestCoefficient
HeestenesStiefelCoefficient
FletcherReevesCoefficient
PolakCoefficient
ConjugateDescentCoefficient
LiuStoreyCoefficient
DaiYuanCoefficient
HagerZhangCoefficient
```
