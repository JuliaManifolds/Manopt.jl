# Optimization with box domains and products of manifolds and boxes

A [`Hyperrectangle`](@extref Manifolds.Hyperrectangle) is, in general, not a manifold but a manifold with corners, thus handling it as a domain in optimization requires special attention.
For simple methods like gradient descent using projected gradient and a stopping criterion involving [`StopWhenProjectedNegativeGradientNormLess`](@ref) may be sufficient, however methods that approximate the Hessian can benefit from a more advanced approach.
The core idea is considering a piecewise quadratic approximation of the objective along the descent direction, and selecting the generalized Cauchy point -- its minimizer.
The points at which the approximation might not be differentiable correspond to hitting new boundaries along the initially selected descent direction.
Then, we can perform standard line search between the initial iterate and the generalized Cauchy point.

Currently `Manopt.jl` can handle domains that are either a [`Hyperrectangle`](@extref Manifolds.Hyperrectangle) or a [`ProductManifold`](@extref ManifoldsBase.ProductManifold) containing a [`Hyperrectangle`](@extref Manifolds.Hyperrectangle) as its first factor and other manifolds as subsequent factors.

## Example

Consider the problem of fitting covariance matrix with box constraints on variance in principal directions.
The objective is log-probability of data under a multivariate normal distribution with zero mean and covariance matrix given by the variable to optimize.
Although there are better ways to solve this problem, expressing it this way allows us to freely extend the objective to more complex scenarios beyond what is possible with closed-form solutions.

First, we set up the problem by generating synthetic data.
The data is sampled from a multivariate normal distribution with known covariance matrix.

```@example example-box-domain
using Manopt, Manifolds, LinearAlgebra, Random, Distributions
using ForwardDiff, DifferentiationInterface, RecursiveArrayTools

N = 5 # dimensionality of data
M_spd = SymmetricPositiveDefinite(N)
M_rot = Rotations(N)
V = rand(M_rot)
cov_matrix = Symmetric(V * Diagonal([0.5; 2.0; 5.0; 10.0; 20.0]) * V')
distr = MvNormal(zeros(N), cov_matrix)
data = Matrix(rand(distr, 200)')  # 200 samples
```

The objective function is defined as follows, with gradient calculated using automatic differentiation.

```@example example-box-domain
function logprob_cost(::AbstractManifold, p)
    D, R = p.x
    logdet = sum(log, D)
    invΣ = R * Diagonal(1 ./ D) * R'
    ll = - 0.5 * size(data, 1) * logdet
    for row in eachrow(data)
        ll -= 0.5 * row' * invΣ * row
    end
    return -ll  # We minimize negative log-likelihood
end

function logprob_gradient(M::AbstractManifold, p)
    Y = DifferentiationInterface.gradient(q -> logprob_cost(M, q), AutoForwardDiff(), p)
    return riemannian_gradient(M, p, Y)
end
```

Finally, we can solve the optimization problem using a quasi-Newton method with box domain support.
We restrict the variances (diagonal elements of the covariance matrix) to be between 1.0 and 100.0.
The covariance matrix is represented using its eigendecomposition $\Sigma = R D R^{\top}$, where $D$ is a diagonal matrix of variances and $R$ is an orthogonal matrix of principal directions.
With constraints on variances, the optimization variable belongs to $[1,100]^N \times \mathrm{SO}(N)$.

```@example example-box-domain
M = ProductManifold(Hyperrectangle(fill(1.0, N), fill(100.0, N)), M_rot)

p0 = ArrayPartition(fill(10.0, N), Matrix{Float64}(I(5)))
p_mle = quasi_Newton(M, logprob_cost, logprob_gradient, p0; stopping_criterion = StopAfterIteration(100) | StopWhenProjectedNegativeGradientNormLess(1e-6))
println("Estimated variances: $(p_mle.x[1])")
cov_matrix_mle = p_mle.x[2] * Diagonal(p_mle.x[1]) * p_mle.x[2]'
println("Estimated covariance matrix:")
println(cov_matrix_mle)
nothing
```

We see that despite the original covariance matrix having variances ranging from 0.5 to 20.0, the estimated covariance matrix respects the box constraints of variances between 1.0 and 100.0.

## Public types and method

```@docs
QuasiNewtonLimitedMemoryBoxDirectionUpdate
```

## Internal types and method

```@docs
Manopt.init_updater!
Manopt.hess_val
Manopt.AbstractFPFPPUpdater
Manopt.GenericFPFPPUpdater
Manopt.get_bounds_index
Manopt.requires_gcp
Manopt.find_generalized_cauchy_point_direction!
Manopt.hess_val_eb
Manopt.LimitedMemoryFPFPPUpdater
Manopt.get_bound_t
Manopt.set_M_current_scale!
Manopt.hess_val_from_wmwt_coords
Manopt.GeneralizedCauchyPointFinder
Manopt.bound_direction_tweak!
```
