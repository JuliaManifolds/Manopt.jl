
@doc raw"""
    HessianProblem <: Problem

specify a problem for hessian based algorithms.

# Fields
* `M`            : a manifold $\mathcal M$
* `cost` : a function $F:\mathcal M→ℝ$ to minimize
* `gradient`     : the gradient $\operatorname{grad}F:\mathcal M
  → \mathcal T\mathcal M$ of the cost function $F$
* `hessian`      : the hessian $\operatorname{Hess}F(x)[⋅]: \mathcal T_{x} \mathcal M
  → \mathcal T_{x} \mathcal M$ of the cost function $F$
* `precon`       : the symmetric, positive definite
    preconditioner (approximation of the inverse of the Hessian of $F$)

# See also
[`truncated_conjugate_gradient_descent`](@ref), [`trust_regions`](@ref)
"""
struct HessianProblem{T,mT<:AbstractManifold,C,G,H,Pre} <: AbstractGradientProblem{T}
    M::mT
    cost::C
    gradient!!::G
    hessian!!::H
    precon::Pre
    function HessianProblem(
        M::mT,
        cost::C,
        grad::G,
        hess::H,
        pre::Pre;
        evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    ) where {mT<:AbstractManifold,C,G,H,Pre}
        return new{typeof(evaluation),mT,C,G,H,Pre}(M, cost, grad, hess, pre)
    end
end

@doc raw"""
    AbstractHessianOptions <: Options

An [`Options`](@ref) type to represent algorithms that employ the Hessian.
These options are assumed to have a field (`gradient`) to store the current gradient ``\operatorname{grad}f(x)``
"""
abstract type AbstractHessianOptions <: AbstractGradientOptions end

@doc raw"""
    TruncatedConjugateGradientOptions <: AbstractHessianOptions

describe the Steihaug-Toint truncated conjugate-gradient method, with

# Fields
a default value is given in brackets if a parameter can be left out in initialization.

* `x` : a point, where the trust-region subproblem needs
    to be solved
* `η` : a tangent vector (called update vector), which solves the
    trust-region subproblem after successful calculation by the algorithm
* `stop` : a [`StoppingCriterion`](@ref).
* `gradient` : the gradient at the current iterate
* `δ` : search direction
* `trust_region_radius` : (`injectivity_radius(M)/4`) the trust-region radius
* `residual` : the gradient
* `randomize` : indicates if the trust-region solve and so the algorithm is to be
        initiated with a random tangent vector. If set to true, no
        preconditioner will be used. This option is set to true in some
        scenarios to escape saddle points, but is otherwise seldom activated.
* `project_vector!` : (`copyto!`) specify a projection operation for tangent vectors
    for numerical stability. A function `(M, Y, p, X) -> ...` working in place of `Y`.
    per default, no projection is perfomed, set it to `project!` to activate projection.

# Constructor

    TruncatedConjugateGradientOptions(M, p, x, η;
        trust_region_radius=injectivity_radius(M)/4,
        randomize=false,
        θ=1.0,
        κ=0.1,
        project_vector! = copyto!,
    )

    and a slightly involved `stopping_criterion`

# See also
[`truncated_conjugate_gradient_descent`](@ref), [`trust_regions`](@ref)
"""
mutable struct TruncatedConjugateGradientOptions{P,T,R<:Real,SC<:StoppingCriterion,Proj} <:
               AbstractHessianOptions
    x::P
    stop::SC
    gradient::T
    η::T
    Hη::T
    δ::T
    Hδ::T
    δHδ::R
    ηPδ::R
    δPδ::R
    ηPη::R
    z::T
    z_r::R
    residual::T
    trust_region_radius::R
    model_value::R
    new_model_value::R
    randomize::Bool
    project!::Proj
    initialResidualNorm::Float64
    function TruncatedConjugateGradientOptions(
        p::HessianProblem,
        x::P,
        η::T,
        trust_region_radius::R,
        randomize::Bool;
        project_vector!::Proj=copyto!,
        θ::Float64=1.0,
        κ::Float64=0.1,
        stop::StoppingCriterion=StopWhenAny(
            StopAfterIteration(manifold_dimension(p.M)),
            StopWhenAll(
                StopIfResidualIsReducedByPower(θ), StopIfResidualIsReducedByFactor(κ)
            ),
            StopWhenTrustRegionIsExceeded(),
            StopWhenCurvatureIsNegative(),
            StopWhenModelIncreased(),
        ),
    ) where {P,T,R<:Real,Proj}
        return TruncatedConjugateGradientOptions(
            p.M,
            x,
            η;
            trust_region_radius=trust_region_radius,
            (project!)=project_vector!,
            randomize=randomize,
            θ=θ,
            κ=κ,
            stopping_criterion=stop,
        )
    end
    function TruncatedConjugateGradientOptions(
        M::AbstractManifold,
        x::P,
        η::T;
        trust_region_radius::R=injectivity_radius(M) / 4.0,
        randomize::Bool=false,
        project!::F=copyto!,
        θ::Float64=1.0,
        κ::Float64=0.1,
        stopping_criterion::StoppingCriterion=StopAfterIteration(manifold_dimension(M)) |
                                              (
                                                  StopIfResidualIsReducedByPower(θ) &
                                                  StopIfResidualIsReducedByFactor(κ)
                                              ) |
                                              StopWhenTrustRegionIsExceeded() |
                                              StopWhenCurvatureIsNegative() |
                                              StopWhenModelIncreased(),
    ) where {P,T,R<:Real,F}
        o = new{P,T,R,typeof(stopping_criterion),F}()
        o.x = x
        o.stop = stopping_criterion
        o.η = η
        o.trust_region_radius = trust_region_radius
        o.randomize = randomize
        o.project! = project!
        o.model_value = zero(trust_region_radius)
        return o
    end
end

@doc raw"""
    TrustRegionsOptions <: AbstractHessianOptions

describe the trust-regions solver, with


# Fields
where all but `x` are keyword arguments in the constructor

* `x` : a point as starting point
* `stop` : (`StopAfterIteration(1000) | StopWhenGradientNormLess(1e-6))
* `trust_region_radius` : the (initial) trust-region radius
* `max_trust_region_radius` : (`sqrt(manifold_dimension(M))`) the maximum trust-region radius
* `randomize` : (`false`) indicates if the trust-region solve is to be initiated with a
        random tangent vector. If set to true, no preconditioner will be
        used. This option is set to true in some scenarios to escape saddle
        points, but is otherwise seldom activated.
* `project!` : (`copyto!`) specify a projection operation for tangent vectors
    for numerical stability. A function `(M, Y, p, X) -> ...` working in place of `Y`.
    per default, no projection is perfomed, set it to `project!` to activate projection.
* `ρ_prime` : (`0.1`) a lower bound of the performance ratio for the iterate that
        decides if the iteration will be accepted or not. If not, the
        trust-region radius will have been decreased. To ensure this,
        ρ'>= 0 must be strictly smaller than 1/4. If ρ' is negative,
        the algorithm is not guaranteed to produce monotonically decreasing
        cost values. It is strongly recommended to set ρ' > 0, to aid
        convergence.
* `ρ_regularization` : (`10000.0`) Close to convergence, evaluating the performance ratio ρ
        is numerically challenging. Meanwhile, close to convergence, the
        quadratic model should be a good fit and the steps should be
        accepted. Regularization lets ρ go to 1 as the model decrease and
        the actual decrease go to zero. Set this option to zero to disable
        regularization (not recommended). When this is not zero, it may happen
        that the iterates produced are not monotonically improving the cost
        when very close to convergence. This is because the corrected cost
        improvement could change sign if it is negative but very small.

# Constructor

    TrustRegionsOptions(M, x)

construct a trust-regions Option with all other fields from above being
keyword arguments

# See also
[`trust_regions`](@ref)
"""
mutable struct TrustRegionsOptions{
    P,T,SC<:StoppingCriterion,RTR<:AbstractRetractionMethod,R<:Real,Proj
} <: AbstractHessianOptions
    x::P
    gradient::T
    stop::SC
    trust_region_radius::R
    max_trust_region_radius::R
    retraction_method::RTR
    randomize::Bool
    project!::Proj
    ρ_prime::R
    ρ_regularization::R

    tcg_options::TruncatedConjugateGradientOptions{P,T,R}

    θ::R
    κ::R

    x_proposal::P
    f_proposal::R

    # Random
    Hgrad::T
    η::T
    Hη::T
    η_Cauchy::T
    Hη_Cauchy::T
    τ::R
    function TrustRegionsOptions{P,T,SC,RTR,R,Proj}(
        x::P,
        grad::T,
        trust_region_radius::R,
        max_trust_region_radius::R,
        ρ_prime::R,
        ρ_regularization::R,
        randomize::Bool,
        stopping_citerion::SC,
        retraction_method::RTR,
        project!::Proj=copyto!,
    ) where {P,T,SC<:StoppingCriterion,RTR<:AbstractRetractionMethod,R<:Real,Proj}
        o = new{P,T,SC,RTR,R,Proj}()
        o.x = x
        o.gradient = grad
        o.stop = stopping_citerion
        o.retraction_method = retraction_method
        o.trust_region_radius = trust_region_radius
        o.max_trust_region_radius = max_trust_region_radius::R
        o.ρ_prime = ρ_prime
        o.ρ_regularization = ρ_regularization
        o.randomize = randomize
        o.project! = project!
        return o
    end
end
@deprecate TrustRegionsOptions(
    x,
    grad,
    trust_region_radius,
    max_trust_region_radius,
    ρ_prime,
    ρ_regularization,
    randomize,
    stopping_criterion;
    retraction_method=ExponentialRetraction(),
    (project_vector!)=copyto!,
) TrustRegionsOptions(
    DefaultManifold(2),
    x;
    gradient=grad,
    ρ_prime=ρ_prime,
    ρ_regularization=ρ_regularization,
    randomize=randomize,
    stopping_criterion=stopping_criterion,
    retraction_method=retraction_method,
    (project!)=project_vector!,
)
function TrustRegionsOptions(
    M::TM,
    x::P;
    gradient::T=zero_vector(M, x),
    ρ_prime::R=0.1,
    ρ_regularization::R=1000.0,
    randomize::Bool=false,
    stopping_criterion::SC=StopAfterIteration(1000) | StopWhenGradientNormLess(1e-6),
    max_trust_region_radius::R=sqrt(manifold_dimension(M)),
    trust_region_radius::R=max_trust_region_radius / 8,
    retraction_method::RTR=default_retraction_method(M),
    project!::Proj=copyto!,
) where {
    TM<:AbstractManifold,
    P,
    T,
    R<:Real,
    SC<:StoppingCriterion,
    RTR<:AbstractRetractionMethod,
    Proj,
}
    return TrustRegionsOptions{P,T,SC,RTR,R,Proj}(
        x,
        gradient,
        trust_region_radius,
        max_trust_region_radius,
        ρ_prime,
        ρ_regularization,
        randomize,
        stopping_criterion,
        retraction_method,
        project!,
    )
end

@doc raw"""
    get_hessian(p::HessianProblem{T}, q, X)
    get_hessian!(p::HessianProblem{T}, Y, q, X)

evaluate the Hessian of a [`HessianProblem`](@ref) `p` at the point `q`
applied to a tangent vector `X`, i.e. ``\operatorname{Hess}f(q)[X]``.

The evaluation is done in place of `Y` for the `!`-variant.
The `T=`[`AllocatingEvaluation`](@ref) problem might still allocate memory within.
When the non-mutating variant is called with a `T=`[`MutatingEvaluation`](@ref)
memory for the result is allocated.
"""
function get_hessian(p::HessianProblem{AllocatingEvaluation}, q, X)
    return p.hessian!!(p.M, q, X)
end
function get_hessian(p::HessianProblem{MutatingEvaluation}, q, X)
    Y = zero_vector(p.M, q)
    return p.hessian!!(p.M, Y, q, X)
end
function get_hessian!(p::HessianProblem{AllocatingEvaluation}, Y, q, X)
    return copyto!(p.M, Y, p.hessian!!(p.M, q, X))
end
function get_hessian!(p::HessianProblem{MutatingEvaluation}, Y, q, X)
    return p.hessian!!(p.M, Y, q, X)
end

@doc raw"""
    get_preconditioner(p,x,ξ)

evaluate the symmetric, positive definite preconditioner (approximation of the
inverse of the Hessian of the cost function `F`) of a
[`HessianProblem`](@ref) `p` at the point `x`applied to a
tangent vector `ξ`.
"""
get_preconditioner(p::HessianProblem, x, X) = p.precon(p.M, x, X)

@doc raw"""
    ApproxHessianFiniteDifference{E, P, T, G, RTR,, VTR, R <: Real}

A functor to approximate the Hessian by a finite difference of gradient evaluation.

Given a point `p` and a direction `X` and the gradient ``\operatorname{grad}F: \mathcal M \to T\mathcal M``
of a function ``F`` the Hessian is approximated as follows:
Let ``c`` be a stepsize, ``X∈ T_p\mathcal M`` a tangent vector and ``q = \operatorname{retr}_p(\frac{c}{\lVert X \rVert_p}X)``
be a step in direction ``X`` of length ``c`` following a retraction
Then we approximate the Hessian by the finite difference of the gradients, where ``\mathcal T_{\cdot\gets\cdot}`` is a vector transport.

```math
\operatorname{Hess}F(p)[X]
 ≈
\frac{\lVert X \rVert_p}{c}\Bigl( \mathcal T_{p\gets q}\bigr(\operatorname{grad}F(q)\bigl) - \operatorname{grad}F(p)\Bigl)
```

 # Fields

* `gradient!!` the gradient function (either allocating or mutating, see `evaluation` parameter)
* `step_length` a step length for the finite difference
* `retraction_method` - a retraction to use
* `vector_transport_method` a vector transport to use

## Internal temporary fields

* `grad_tmp` a temporary storage for the gradient at the current `p`
* `grad_dir_tmp` a temporary storage for the gradient at the current `p_dir`
* `p_dir::P` a temporary storage to the forward direction (i.e. ``q`` above)

# Constructor

    ApproximateFiniteDifference(M, p, gradF; kwargs...)

## Keyword arguments

* `evaluation` (`AllocatingEvaluation()`) whether the gradient is given as an allocation function or an in-place (`MutatingEvaluation()`).
* `steplength` (``2^{-14}``) step length ``c`` to approximate the gradient evaluations
* `retraction_method` – (`default_retraction_method(M)`) a `retraction(M, p, X)` to use in the approximation.
* `vector_transport_method` - (`default_vector_transport_method(M)`) a vector transport to use
"""
mutable struct ApproxHessianFiniteDifference{E,P,T,G,RTR,VTR,R<:Real}
    p_dir::P
    gradient!!::G
    grad_tmp::T
    grad_tmp_dir::T
    retraction_method::RTR
    vector_transport_method::VTR
    steplength::R
end
function ApproxHessianFiniteDifference(
    M::mT,
    p::P,
    grad::G;
    steplength::R=2^-14,
    evaluation=AllocatingEvaluation(),
    retraction_method::RTR=default_retraction_method(M),
    vector_transport_method::VTR=default_vector_transport_method(M),
) where {
    mT<:AbstractManifold,
    P,
    G,
    R<:Real,
    RTR<:AbstractRetractionMethod,
    VTR<:AbstractVectorTransportMethod,
}
    X = zero_vector(M, p)
    Y = zero_vector(M, p)
    return ApproxHessianFiniteDifference{typeof(evaluation),P,typeof(X),G,RTR,VTR,R}(
        p, grad, X, Y, retraction_method, vector_transport_method, steplength
    )
end

function (f::ApproxHessianFiniteDifference{AllocatingEvaluation})(M, p, X)
    norm_X = norm(M, p, X)
    (norm_X ≈ zero(norm_X)) && return zero_vector(M, p)
    c = f.steplength / norm_X
    f.grad_tmp .= f.gradient!!(M, p)
    f.p_dir .= retract(M, p, c * X, f.retraction_method)
    f.grad_tmp_dir .= f.gradient!!(M, f.p_dir)
    f.grad_tmp_dir .= vector_transport_to(
        M, f.p_dir, f.grad_tmp_dir, p, f.vector_transport_method
    )
    return (1 / c) * (f.grad_tmp_dir - f.grad_tmp)
end

function (f::ApproxHessianFiniteDifference{MutatingEvaluation})(M, Y, p, X)
    norm_X = norm(M, p, X)
    (norm_X ≈ zero(norm_X)) && return zero_vector!(M, X, p)
    c = f.steplength / norm_X
    f.gradient!!(M, f.grad_tmp, p)
    retract!(M, f.p_dir, p, c * X, f.retraction_method)
    f.gradient!!(M, f.grad_tmp_dir, f.p_dir)
    vector_transport_to!(
        M, f.grad_tmp_dir, f.p_dir, f.grad_tmp_dir, p, f.vector_transport_method
    )
    Y .= (1 / c) .* (f.grad_tmp_dir .- f.grad_tmp)
    return Y
end

@doc raw"""
    ApproxHessianSymmetricRankOne{E, P, G, T, B<:AbstractBasis{ℝ}, VTR, R<:Real}

A functor to approximate the Hessian by the symmetric rank one update.

# Fields

* `gradient!!` the gradient function (either allocating or mutating, see `evaluation` parameter).
* `ν` a small real number to ensure that the denominator in the update does not become too small and thus the method does not break down.
* `vector_transport_method` a vector transport to use.

## Internal temporary fields

* `p_tmp` a temporary storage the current point `p`.
* `grad_tmp` a temporary storage for the gradient at the current `p`.
* `matrix` a temporary storage for the matrix representation of the approximating operator.
* `basis` a temporary storage for an orthonormal basis at the current `p`.

# Constructor

    ApproxHessianSymmetricRankOne(M, p, gradF; kwargs...)

## Keyword arguments

* `initial_operator` (`Matrix{Float64}(I, manifold_dimension(M), manifold_dimension(M))`) the matrix representation of the initial approximating operator.
* `basis` (`DefaultOrthonormalBasis()`) an orthonormal basis in the tangent space of the initial iterate p. 
* `nu` (`-1`) 
* `evaluation` (`AllocatingEvaluation()`) whether the gradient is given as an allocation function or an in-place (`MutatingEvaluation()`).
* `vector_transport_method` (`ParallelTransport()`) vector transport ``\mathcal T_{\cdot\gets\cdot}`` to use.
"""
mutable struct ApproxHessianSymmetricRankOne{E,P,G,T,B<:AbstractBasis{ℝ},VTR,R<:Real}
    p_tmp::P
    gradient!!::G
    grad_tmp::T
    matrix::Matrix
    basis::B
    vector_transport_method::VTR
    ν::R
end
function ApproxHessianSymmetricRankOne(
    M::mT,
    p::P,
    gradient::G;
    initial_operator::AbstractMatrix=Matrix{Float64}(
        I, manifold_dimension(M), manifold_dimension(M)
    ),
    basis::B=DefaultOrthonormalBasis(),
    nu::R=-1.0,
    evaluation=AllocatingEvaluation(),
    vector_transport_method::VTR=ParallelTransport(),
) where {mT<:AbstractManifold,P,G,B<:AbstractBasis{ℝ},R<:Real,VTR<:AbstractVectorTransportMethod}
    grad_tmp = gradient(M, p)
    return ApproxHessianSymmetricRankOne{typeof(evaluation),P,G,typeof(grad_tmp),B,VTR,R}(
        p, gradient, grad_tmp, initial_operator, basis, vector_transport_method, nu
    )
end

function (f::ApproxHessianSymmetricRankOne{AllocatingEvaluation})(M, p, X)
    # Update Basis if necessary
    if p != f.p_tmp
        update_basis!(f.basis, M, f.p_tmp, p, f.vector_transport_method)
        copyto!(f.p_tmp, p)
        f.grad_tmp = f.gradient!!(M, f.p_tmp)
    end

    # Apply Hessian approximation on vector
    return get_vector(
        M, f.p_tmp, f.matrix * get_coordinates(M, f.p_tmp, X, f.basis), f.basis
    )
end

function (f::ApproxHessianSymmetricRankOne{MutatingEvaluation})(M, Y, p, X)
    # Update Basis if necessary
    # if distance(M, p, f.p_tmp) >= eps(Float64)
    if p != f.p_tmp
        update_basis!(f.basis, M, f.p_tmp, p, f.vector_transport_method)
        copyto!(f.p_tmp, p)
        f.grad_tmp = f.gradient!!(M, f.p_tmp)
    end

    # Apply Hessian approximation on vector
    Y .= get_vector(M, f.p_tmp, f.matrix * get_coordinates(M, f.p_tmp, X, f.basis), f.basis)

    return Y
end

function update_hessian!(M, f::ApproxHessianSymmetricRankOne, p, p_proposal, X)
    yk_c = get_coordinates(
        M,
        p,
        vector_transport_to(
            M, p_proposal, f.gradient!!(M, p_proposal), p, f.vector_transport_method
        ) - f.grad_tmp,
        f.basis,
    )
    sk_c = get_coordinates(M, p, X, f.basis)
    srvec = yk_c - f.matrix * sk_c
    if f.ν < 0 || abs(dot(srvec, sk_c)) >= f.ν * norm(srvec) * norm(sk_c)
        f.matrix = f.matrix + srvec * srvec' / (srvec' * sk_c)
    end
end

function update_hessian_basis!(M, f::ApproxHessianSymmetricRankOne, p)
    update_basis!(f.basis, M, f.p_tmp, p, f.vector_transport_method)
    copyto!(f.p_tmp, p)
    return f.grad_tmp = f.gradient!!(M, f.p_tmp)
end

function update_hessian!(M, f, p, p_proposal, X) end

function update_hessian_basis!(M, f, p) end

@doc raw"""
    ApproxHessianBFGS{E, P, G, T, B<:AbstractBasis{ℝ}, VTR, R<:Real}

A functor to approximate the Hessian by the BFGS update.

# Fields

* `gradient!!` the gradient function (either allocating or mutating, see `evaluation` parameter).
* `scale` 
* `vector_transport_method` a vector transport to use.

## Internal temporary fields

* `p_tmp` a temporary storage the current point `p`.
* `grad_tmp` a temporary storage for the gradient at the current `p`.
* `matrix` a temporary storage for the matrix representation of the approximating operator.
* `basis` a temporary storage for an orthonormal basis at the current `p`.

# Constructor

    ApproxHessianBFGS(M, p, gradF; kwargs...)

## Keyword arguments

* `initial_operator` (`Matrix{Float64}(I, manifold_dimension(M), manifold_dimension(M))`) the matrix representation of the initial approximating operator.
* `basis` (`DefaultOrthonormalBasis()`) an orthonormal basis in the tangent space of the initial iterate p. 
* `nu` (`-1`) 
* `evaluation` (`AllocatingEvaluation()`) whether the gradient is given as an allocation function or an in-place (`MutatingEvaluation()`).
* `vector_transport_method` (`ParallelTransport()`) vector transport ``\mathcal T_{\cdot\gets\cdot}`` to use.
"""
mutable struct ApproxHessianBFGS{
    E,P,G,T,B<:AbstractBasis{ℝ},VTR<:AbstractVectorTransportMethod
}
    p_tmp::P
    gradient!!::G
    grad_tmp::T
    matrix::Matrix
    basis::B
    vector_transport_method::VTR
    scale::Bool
end
function ApproxHessianBFGS(
    M::mT,
    p::P,
    gradient::G;
    initial_operator::AbstractMatrix=Matrix{Float64}(
        I, manifold_dimension(M), manifold_dimension(M)
    ),
    basis::B=DefaultOrthonormalBasis(),
    scale::Bool=true,
    evaluation=AllocatingEvaluation(),
    vector_transport_method::VTR=ParallelTransport(),
) where {mT<:AbstractManifold,P,G,B<:AbstractBasis{ℝ},VTR<:AbstractVectorTransportMethod}
    grad_tmp = gradient(M, p)
    return ApproxHessianBFGS{typeof(evaluation),P,G,typeof(grad_tmp),B,VTR}(
        p, gradient, grad_tmp, initial_operator, basis, vector_transport_method, scale
    )
end

function (f::ApproxHessianBFGS{AllocatingEvaluation})(M, p, X)
    # Update Basis if necessary
    if p != f.p_tmp
        update_basis!(f.basis, M, f.p_tmp, p, f.vector_transport_method)
        copyto!(f.p_tmp, p)
        f.grad_tmp = f.gradient!!(M, f.p_tmp)
    end

    # Apply Hessian approximation on vector
    return get_vector(
        M, f.p_tmp, f.matrix * get_coordinates(M, f.p_tmp, X, f.basis), f.basis
    )
end

function (f::ApproxHessianBFGS{MutatingEvaluation})(M, Y, p, X)
    # Update Basis if necessary
    if p != f.p_tmp
        update_basis!(f.basis, M, f.p_tmp, p, f.vector_transport_method)
        copyto!(f.p_tmp, p)
        f.grad_tmp = f.gradient!!(M, f.p_tmp)
    end

    # Apply Hessian approximation on vector
    Y .= get_vector(M, f.p_tmp, f.matrix * get_coordinates(M, f.p_tmp, X, f.basis), f.basis)

    return Y
end

function update_hessian!(M, f::ApproxHessianBFGS, p, p_proposal, X)
    yk_c = get_coordinates(
        M,
        p,
        vector_transport_to(
            M, p_proposal, f.gradient!!(M, p_proposal), p, f.vector_transport_method
        ) - f.grad_tmp,
        f.basis,
    )
    sk_c = get_coordinates(M, p, X, f.basis)
    skyk_c = dot(sk_c, yk_c)
    return f.matrix =
        f.matrix + yk_c * yk_c' / skyk_c -
        f.matrix * sk_c * sk_c' * f.matrix / dot(sk_c, f.matrix * sk_c)
end

function update_hessian_basis!(M, f::ApproxHessianBFGS, p)
    update_basis!(f.basis, M, f.p_tmp, p, f.vector_transport_method)
    copyto!(f.p_tmp, p)
    return f.grad_tmp = f.gradient!!(M, f.p_tmp)
end

@doc raw"""
    StopIfResidualIsReducedByFactor <: StoppingCriterion

A functor for testing if the norm of residual at the current iterate is reduced
by a factor compared to the norm of the initial residual, i.e.
$\Vert r_k \Vert_x \leqq κ \Vert r_0 \Vert_x$.
In this case the algorithm reached linear convergence.

# Fields
* `κ` – the reduction factor
* `initialResidualNorm` - stores the norm of the residual at the initial vector
    ``η`` of the Steihaug-Toint tcg mehtod [`truncated_conjugate_gradient_descent`](@ref)
* `reason` – stores a reason of stopping if the stopping criterion has one be
  reached, see [`get_reason`](@ref).

# Constructor

    StopIfResidualIsReducedByFactor(iRN, κ)

initialize the StopIfResidualIsReducedByFactor functor to indicate to stop after
the norm of the current residual is lesser than the norm of the initial residual
iRN times κ.

# See also
[`truncated_conjugate_gradient_descent`](@ref), [`trust_regions`](@ref)
"""
mutable struct StopIfResidualIsReducedByFactor <: StoppingCriterion
    κ::Float64
    reason::String
    StopIfResidualIsReducedByFactor(κ::Float64) = new(κ, "")
end
function (c::StopIfResidualIsReducedByFactor)(
    p::P, o::O, i::Int
) where {P<:HessianProblem,O<:TruncatedConjugateGradientOptions}
    if norm(p.M, o.x, o.residual) <= o.initialResidualNorm * c.κ && i > 0
        c.reason = "The algorithm reached linear convergence (residual at least reduced by κ=$(c.κ)).\n"
        return true
    end
    return false
end

@doc raw"""
    StopIfResidualIsReducedByPower <: StoppingCriterion

A functor for testing if the norm of residual at the current iterate is reduced
by a power of 1+θ compared to the norm of the initial residual, i.e.
$\Vert r_k \Vert_x \leqq  \Vert r_0 \Vert_{x}^{1+\theta}$. In this case the
algorithm reached superlinear convergence.

# Fields
* `θ` – part of the reduction power
* `initialResidualNorm` - stores the norm of the residual at the initial vector
    $η$ of the Steihaug-Toint tcg mehtod [`truncated_conjugate_gradient_descent`](@ref)
* `reason` – stores a reason of stopping if the stopping criterion has one be
    reached, see [`get_reason`](@ref).

# Constructor

    StopIfResidualIsReducedByPower(iRN, θ)

initialize the StopIfResidualIsReducedByFactor functor to indicate to stop after
the norm of the current residual is lesser than the norm of the initial residual
iRN to the power of 1+θ.

# See also
[`truncated_conjugate_gradient_descent`](@ref), [`trust_regions`](@ref)
"""
mutable struct StopIfResidualIsReducedByPower <: StoppingCriterion
    θ::Float64
    reason::String
    StopIfResidualIsReducedByPower(θ::Float64) = new(θ, "")
end
function (c::StopIfResidualIsReducedByPower)(
    p::P, o::O, i::Int
) where {P<:HessianProblem,O<:TruncatedConjugateGradientOptions}
    if norm(p.M, o.x, o.residual) <= o.initialResidualNorm^(1 + c.θ) && i > 0
        c.reason = "The algorithm reached superlinear convergence (residual at least reduced by power 1 + θ=$(1+(c.θ))).\n"
        return true
    end
    return false
end
@doc raw"""
    update_stopping_criterion!(c::StopIfResidualIsReducedByPower, :ResidualPower, v)

Update the residual Power ``θ`` time period after which an algorithm shall stop.
"""
function update_stopping_criterion!(
    c::StopIfResidualIsReducedByPower, ::Val{:ResidualPower}, v
)
    c.θ = v
    return c
end
@doc raw"""
    StopWhenTrustRegionIsExceeded <: StoppingCriterion

A functor for testing if the norm of the next iterate in the  Steihaug-Toint tcg
mehtod is larger than the trust-region radius, i.e. $\Vert η_{k}^{*} \Vert_x
≧ trust_region_radius$. terminate the algorithm when the trust region has been left.

# Fields
* `reason` – stores a reason of stopping if the stopping criterion has one be
    reached, see [`get_reason`](@ref).
* `storage` – stores the necessary parameters `η, δ, residual` to check the
    criterion.

# Constructor

    StopWhenTrustRegionIsExceeded([a])

initialize the StopWhenTrustRegionIsExceeded functor to indicate to stop after
the norm of the next iterate is greater than the trust-region radius using the
[`StoreOptionsAction`](@ref) `a`, which is initialized to store
`:η, :δ, :residual` by default.

# See also
[`truncated_conjugate_gradient_descent`](@ref), [`trust_regions`](@ref)
"""
mutable struct StopWhenTrustRegionIsExceeded <: StoppingCriterion
    reason::String
end
StopWhenTrustRegionIsExceeded() = StopWhenTrustRegionIsExceeded("")
function (c::StopWhenTrustRegionIsExceeded)(
    ::HessianProblem, o::TruncatedConjugateGradientOptions, i::Int
)
    if o.ηPη >= o.trust_region_radius^2 && i >= 0
        c.reason = "Trust-region radius violation (‖η‖² = $(o.ηPη)) >= $(o.trust_region_radius^2) = trust_region_radius²). \n"
        return true
    end
    return false
end

@doc raw"""
    StopWhenCurvatureIsNegative <: StoppingCriterion

A functor for testing if the curvature of the model is negative, i.e.
$\langle \delta_k, \operatorname{Hess}[F](\delta_k)\rangle_x \leqq 0$.
In this case, the model is not strictly convex, and the stepsize as computed
does not give a reduction of the model.

# Fields
* `reason` – stores a reason of stopping if the stopping criterion has one be
    reached, see [`get_reason`](@ref).

# Constructor

    StopWhenCurvatureIsNegative()

# See also
[`truncated_conjugate_gradient_descent`](@ref), [`trust_regions`](@ref)
"""
mutable struct StopWhenCurvatureIsNegative <: StoppingCriterion
    reason::String
end
StopWhenCurvatureIsNegative() = StopWhenCurvatureIsNegative("")
function (c::StopWhenCurvatureIsNegative)(
    ::HessianProblem, o::TruncatedConjugateGradientOptions, i::Int
)
    if o.δHδ <= 0 && i > 0
        c.reason = "Negative curvature. The model is not strictly convex (⟨δ,Hδ⟩_x = $(o.δHδ))) <= 0).\n"
        return true
    end
    return false
end

@doc raw"""
    StopWhenModelIncreased <: StoppingCriterion

A functor for testing if the curvature of the model value increased.

# Fields
* `reason` – stores a reason of stopping if the stopping criterion has one be
    reached, see [`get_reason`](@ref).

# Constructor

    StopWhenModelIncreased()

# See also
[`truncated_conjugate_gradient_descent`](@ref), [`trust_regions`](@ref)
"""
mutable struct StopWhenModelIncreased <: StoppingCriterion
    reason::String
end
StopWhenModelIncreased() = StopWhenModelIncreased("")
function (c::StopWhenModelIncreased)(
    ::HessianProblem, o::TruncatedConjugateGradientOptions, i::Int
)
    if i > 0 && (o.new_model_value > o.model_value)
        c.reason = "Model value increased from $(o.model_value) to $(o.new_model_value).\n"
        return true
    end
    return false
end
