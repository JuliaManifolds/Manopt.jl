@doc """
    AbstractHessianSolverState <: AbstractGradientSolverState

An [`AbstractManoptSolverState`](@ref) type to represent algorithms that employ the Hessian.
These options are assumed to have a field (`gradient`) to store the current gradient ``$(_tex(:grad))f(x)``
"""
abstract type AbstractHessianSolverState <: AbstractGradientSolverState end

"""
    AbstractManifoldHessianObjective{E<:AbstractEvaluationType,F, G, H} <: AbstractManifoldFirstOrderObjective{E,Tuple{F,G}}

An abstract type for all objectives that provide a (full) Hessian, where
`T` is a [`AbstractEvaluationType`](@ref) for the gradient and Hessian functions.
"""
abstract type AbstractManifoldHessianObjective{E <: AbstractEvaluationType, F, G, H} <:
AbstractManifoldFirstOrderObjective{E, Tuple{F, G}} end

@doc """
    ManifoldHessianObjective{T<:AbstractEvaluationType,C,G,H,Pre} <: AbstractManifoldHessianObjective{T,C,G,H}

specify a problem for Hessian based algorithms.

# Fields

* `cost`:           a function ``f:$(_math(:M))→ℝ`` to minimize
* `gradient`:       the gradient ``$(_tex(:grad))f:$(_math(:M)) → $(_math(:TM))`` of the cost function ``f``
* `hessian`:        the Hessian ``$(_tex(:Hess))f(x)[⋅]: $(_math(:TpM; p = "x")) → $(_math(:TpM; p = "x"))`` of the cost function ``f``
* `preconditioner`: the symmetric, positive definite preconditioner
  as an approximation of the inverse of the Hessian of ``f``, a map with the same
  input variables as the `hessian` to numerically stabilize iterations when the Hessian is
  ill-conditioned

Depending on the [`AbstractEvaluationType`](@ref) `T` the gradient and can have to forms

* as a function `(M, p) -> X`  and `(M, p, X) -> Y`, resp., an [`AllocatingEvaluation`](@ref)
* as a function `(M, X, p) -> X` and (M, Y, p, X), resp., an [`InplaceEvaluation`](@ref)

# Constructor
    ManifoldHessianObjective(f, grad_f, Hess_f, preconditioner = (M, p, X) -> X;
        evaluation=AllocatingEvaluation())

# See also

[`truncated_conjugate_gradient_descent`](@ref), [`trust_regions`](@ref)
"""
struct ManifoldHessianObjective{T <: AbstractEvaluationType, C, G, H, Pre} <:
    AbstractManifoldHessianObjective{T, C, G, H}
    cost::C
    gradient!!::G
    hessian!!::H
    preconditioner!!::Pre
    function ManifoldHessianObjective(
            cost::C,
            grad::G,
            hess::H,
            precond = nothing;
            evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        ) where {C, G, H}
        if isnothing(precond)
            if evaluation isa InplaceEvaluation
                precond = (M, Y, p, X) -> (Y .= X)
            else
                precond = (M, p, X) -> X
            end
        end
        return new{typeof(evaluation), C, G, H, typeof(precond)}(cost, grad, hess, precond)
    end
end

function get_gradient(
        M::AbstractManifold, mho::ManifoldHessianObjective{AllocatingEvaluation}, p
    )
    return mho.gradient!!(M, p)
end
function get_gradient(
        M::AbstractManifold, mho::ManifoldHessianObjective{InplaceEvaluation}, p
    )
    X = zero_vector(M, p)
    mho.gradient!!(M, X, p)
    return X
end
function get_gradient!(
        M::AbstractManifold, X, mho::ManifoldHessianObjective{AllocatingEvaluation}, p
    )
    copyto!(M, X, p, mho.gradient!!(M, p))
    return X
end
function get_gradient!(
        M::AbstractManifold, Y, mho::ManifoldHessianObjective{InplaceEvaluation}, p
    )
    return mho.gradient!!(M, Y, p)
end

function get_gradient_function(mho::ManifoldHessianObjective, recursive = false)
    return mho.gradient!!
end

@doc """
    Y = get_hessian(amp::AbstractManoptProblem{T}, p, X)
    get_hessian!(amp::AbstractManoptProblem{T}, Y, p, X)

evaluate the Hessian of an [`AbstractManoptProblem`](@ref) `amp` at `p`
applied to a tangent vector `X`, computing ``$(_tex(:Hess))f(q)[X]``,
which can also happen in-place of `Y`.
"""
function get_hessian(amp::AbstractManoptProblem, p, X)
    return get_hessian(get_manifold(amp), get_objective(amp), p, X)
end
function get_hessian!(amp::AbstractManoptProblem, Y, p, X)
    return get_hessian!(get_manifold(amp), Y, get_objective(amp), p, X)
end

function get_hessian(M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p, X)
    return get_hessian(M, get_objective(admo, false), p, X)
end
function get_hessian(
        M::AbstractManifold, mho::ManifoldHessianObjective{AllocatingEvaluation}, p, X
    )
    return mho.hessian!!(M, p, X)
end
function get_hessian(
        M::AbstractManifold, mho::ManifoldHessianObjective{InplaceEvaluation}, p, X
    )
    Y = zero_vector(M, p)
    mho.hessian!!(M, Y, p, X)
    return Y
end
function get_hessian!(
        M::AbstractManifold, Y, admo::AbstractDecoratedManifoldObjective, p, X
    )
    return get_hessian!(M, Y, get_objective(admo, false), p, X)
end
function get_hessian!(
        M::AbstractManifold, Y, mho::ManifoldHessianObjective{AllocatingEvaluation}, p, X
    )
    copyto!(M, Y, p, mho.hessian!!(M, p, X))
    return Y
end
function get_hessian!(
        M::AbstractManifold, Y, mho::ManifoldHessianObjective{InplaceEvaluation}, p, X
    )
    mho.hessian!!(M, Y, p, X)
    return Y
end

@doc """
    get_hessian_function(amgo::ManifoldHessianObjective{E<:AbstractEvaluationType})

return the function to evaluate (just) the Hessian ``$(_tex(:Hess)) f(p)``.
Depending on the [`AbstractEvaluationType`](@ref) `E` this is a function

* `(M, p, X) -> Y` for the [`AllocatingEvaluation`](@ref) case
* `(M, Y, p, X) -> X` for the [`InplaceEvaluation`](@ref), working in-place of `Y`.
"""
get_hessian_function(mho::ManifoldHessianObjective, recursive::Bool = false) = mho.hessian!!
function get_hessian_function(
        admo::AbstractDecoratedManifoldObjective, recursive::Bool = false
    )
    return get_hessian_function(get_objective(admo, recursive))
end

@doc """
    get_preconditioner(amp::AbstractManoptProblem, p, X)

evaluate the symmetric, positive definite preconditioner (approximation of the
inverse of the Hessian of the cost function `f`) of a
[`AbstractManoptProblem`](@ref) `amp`s objective at the point `p` applied to a
tangent vector `X`.
"""
function get_preconditioner(amp::AbstractManoptProblem, p, X)
    return get_preconditioner(get_manifold(amp), get_objective(amp), p, X)
end
function get_preconditioner!(amp::AbstractManoptProblem, Y, p, X)
    return get_preconditioner!(get_manifold(amp), Y, get_objective(amp), p, X)
end

@doc """
    get_preconditioner(M::AbstractManifold, mho::ManifoldHessianObjective, p, X)

evaluate the symmetric, positive definite preconditioner (approximation of the
inverse of the Hessian of the cost function `F`) of a
[`ManifoldHessianObjective`](@ref) `mho` at the point `p` applied to a
tangent vector `X`.
"""
function get_preconditioner(
        M::AbstractManifold, mho::ManifoldHessianObjective{AllocatingEvaluation}, p, X
    )
    return mho.preconditioner!!(M, p, X)
end
function get_preconditioner(
        M::AbstractManifold, mho::ManifoldHessianObjective{InplaceEvaluation}, p, X
    )
    Y = zero_vector(M, p)
    mho.preconditioner!!(M, Y, p, X)
    return Y
end
function get_preconditioner(
        M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p, X
    )
    return get_preconditioner(M, get_objective(admo, false), p, X)
end

function get_preconditioner!(
        M::AbstractManifold, Y, mho::ManifoldHessianObjective{AllocatingEvaluation}, p, X
    )
    copyto!(M, Y, p, mho.preconditioner!!(M, p, X))
    return Y
end
function get_preconditioner!(
        M::AbstractManifold, Y, admo::AbstractDecoratedManifoldObjective, p, X
    )
    return get_preconditioner!(M, Y, get_objective(admo, false), p, X)
end
function get_preconditioner!(
        M::AbstractManifold, Y, mho::ManifoldHessianObjective{InplaceEvaluation}, p, X
    )
    mho.preconditioner!!(M, Y, p, X)
    return Y
end

update_hessian!(M, f, p, p_proposal, X) = f

update_hessian_basis!(M, f, p) = f

@doc """
    AbstractApproxHessian <: Function

An abstract supertype for approximate Hessian functions, declares them also to be functions.
"""
abstract type AbstractApproxHessian <: Function end

_doc_ApproxHessian_formula = raw"""
```math
$(_tex(:Hess))f(p)[X] ≈
\frac{\lVert X \rVert_p}{c}\Bigl(
  \mathcal T_{p\gets q}\bigr($(_tex(:grad))f(q)\bigl) - $(_tex(:grad))f(p)
\Bigl)
```
"""
_doc_ApproxHessian_step = raw"\operatorname{retr}_p(\frac{c}{\lVert X \rVert_p}X)"

@doc """
    ApproxHessianFiniteDifference{E, P, T, G, RTR, VTR, R <: Real} <: AbstractApproxHessian

A functor to approximate the Hessian by a finite difference of gradient evaluation.

Given a point `p` and a direction `X` and the gradient ``$(_tex(:grad)) f(p)``
of a function ``f`` the Hessian is approximated as follows:
let ``c`` be a stepsize, ``X ∈ $(_math(:TpM))`` a tangent vector and ``q = $_doc_ApproxHessian_step``
be a step in direction ``X`` of length ``c`` following a retraction
Then the Hessian is approximated by the finite difference of the gradients,
where ``$(_math(:vector_transport, :symbol))`` is a vector transport.

$_doc_ApproxHessian_formula

 # Fields

* `gradient!!`:              the gradient function (either allocating or mutating, see `evaluation` parameter)
* `step_length`:             a step length for the finite difference
$(_var(:Keyword, :retraction_method))
$(_var(:Keyword, :vector_transport_method))

## Internal temporary fields

* `grad_tmp`:     a temporary storage for the gradient at the current `p`
* `grad_dir_tmp`: a temporary storage for the gradient at the current `p_dir`
* `p_dir::P`:     a temporary storage to the forward direction (or the ``q`` in the formula)

# Constructor

    ApproximateFiniteDifference(M, p, grad_f; kwargs...)

## Keyword arguments

$(_var(:Keyword, :evaluation))
* `steplength=`2^{-14}``: step length ``c`` to approximate the gradient evaluations
$(_var(:Keyword, :retraction_method))
$(_var(:Keyword, :vector_transport_method))
"""
mutable struct ApproxHessianFiniteDifference{E, P, T, G, RTR, VTR, R <: Real} <:
    AbstractApproxHessian
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
        grad_f::G;
        tangent_vector = zero_vector(M, p),
        steplength::R = 2^-14,
        evaluation = AllocatingEvaluation(),
        retraction_method::RTR = default_retraction_method(M, typeof(p)),
        vector_transport_method::VTR = default_vector_transport_method(M, typeof(p)),
    ) where {
        mT <: AbstractManifold,
        P,
        G,
        R <: Real,
        RTR <: AbstractRetractionMethod,
        VTR <: AbstractVectorTransportMethod,
    }
    X = copy(M, p, tangent_vector)
    Y = copy(M, p, tangent_vector)
    return ApproxHessianFiniteDifference{typeof(evaluation), P, typeof(X), G, RTR, VTR, R}(
        p, grad_f, X, Y, retraction_method, vector_transport_method, steplength
    )
end

function (f::ApproxHessianFiniteDifference{AllocatingEvaluation})(M, p, X)
    norm_X = norm(M, p, X)
    (norm_X ≈ zero(norm_X)) && return zero_vector(M, p)
    c = f.steplength / norm_X
    f.grad_tmp .= f.gradient!!(M, p)
    retract!(M, f.p_dir, p, c * X, f.retraction_method)
    f.grad_tmp_dir .= f.gradient!!(M, f.p_dir)
    vector_transport_to!(
        M, f.grad_tmp_dir, f.p_dir, f.grad_tmp_dir, p, f.vector_transport_method
    )
    return (1 / c) * (f.grad_tmp_dir - f.grad_tmp)
end
function (f::ApproxHessianFiniteDifference{InplaceEvaluation})(M, Y, p, X)
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

@doc """
    ApproxHessianSymmetricRankOne{E, P, G, T, B<:AbstractBasis{ℝ}, VTR, R<:Real} <: AbstractApproxHessian

A functor to approximate the Hessian by the symmetric rank one update.

# Fields

* `gradient!!`: the gradient function (either allocating or mutating, see `evaluation` parameter).
* `ν`: a small real number to ensure that the denominator in the update does not become too small and thus the method does not break down.
$(_var(:Keyword, :vector_transport_method)).

## Internal temporary fields

* `p_tmp`: a temporary storage the current point `p`.
* `grad_tmp`: a temporary storage for the gradient at the current `p`.
* `matrix`: a temporary storage for the matrix representation of the approximating operator.
* `basis`: a temporary storage for an orthonormal basis at the current `p`.

# Constructor

    ApproxHessianSymmetricRankOne(M, p, gradF; kwargs...)

## Keyword arguments

* `initial_operator` (`Matrix{Float64}(I, manifold_dimension(M), manifold_dimension(M))`) the matrix representation of the initial approximating operator.
* `basis` (`DefaultOrthonormalBasis()`) an orthonormal basis in the tangent space of the initial iterate p.
* `nu` (`-1`)
$(_var(:Keyword, :evaluation))
$(_var(:Keyword, :vector_transport_method))
"""
mutable struct ApproxHessianSymmetricRankOne{E, P, G, T, B <: AbstractBasis{ℝ}, VTR, R <: Real} <:
    AbstractApproxHessian
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
        initial_operator::AbstractMatrix = Matrix{Float64}(
            I, manifold_dimension(M), manifold_dimension(M)
        ),
        basis::B = default_basis(M, typeof(p)),
        nu::R = -1.0,
        evaluation = AllocatingEvaluation(),
        vector_transport_method::VTM = default_vector_transport_method(M, typeof(p)),
    ) where {
        mT <: AbstractManifold, P, G, B <: AbstractBasis{ℝ}, R <: Real, VTM <: AbstractVectorTransportMethod,
    }
    if evaluation isa AllocatingEvaluation
        grad_tmp = gradient(M, p)
    elseif evaluation isa InplaceEvaluation
        grad_tmp = zero_vector(M, p)
        gradient(M, grad_tmp, p)
    end

    return ApproxHessianSymmetricRankOne{typeof(evaluation), P, G, typeof(grad_tmp), B, VTM, R}(
        p, gradient, grad_tmp, initial_operator, basis, vector_transport_method, nu
    )
end

function (f::ApproxHessianSymmetricRankOne{AllocatingEvaluation})(M, p, X)
    # Update Basis if necessary
    if p != f.p_tmp
        update_basis!(f.basis, M, f.p_tmp, p, f.vector_transport_method)
        copyto!(M, f.p_tmp, p)
        f.grad_tmp = f.gradient!!(M, f.p_tmp)
    end

    # Apply Hessian approximation on vector
    return get_vector(
        M, f.p_tmp, f.matrix * get_coordinates(M, f.p_tmp, X, f.basis), f.basis
    )
end

function (f::ApproxHessianSymmetricRankOne{InplaceEvaluation})(M, Y, p, X)
    # Update Basis if necessary
    if p != f.p_tmp
        update_basis!(f.basis, M, f.p_tmp, p, f.vector_transport_method)
        copyto!(f.p_tmp, p)
        f.gradient!!(M, f.grad_tmp, f.p_tmp)
    end

    # Apply Hessian approximation on vector
    Y .= get_vector(M, f.p_tmp, f.matrix * get_coordinates(M, f.p_tmp, X, f.basis), f.basis)

    return Y
end

function update_hessian!(
        M, f::ApproxHessianSymmetricRankOne{AllocatingEvaluation}, p, p_proposal, X
    )
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
    return if f.ν < 0 || abs(dot(srvec, sk_c)) >= f.ν * norm(srvec) * norm(sk_c)
        f.matrix = f.matrix + srvec * srvec' / (srvec' * sk_c)
    end
end

function update_hessian!(
        M::AbstractManifold,
        f::ApproxHessianSymmetricRankOne{InplaceEvaluation},
        p,
        p_proposal,
        X,
    )
    grad_proposal = zero_vector(M, p_proposal)
    f.gradient!!(M, grad_proposal, p_proposal)
    yk_c = get_coordinates(
        M,
        p,
        vector_transport_to(M, p_proposal, grad_proposal, p, f.vector_transport_method) -
            f.grad_tmp,
        f.basis,
    )
    sk_c = get_coordinates(M, p, X, f.basis)
    srvec = yk_c - f.matrix * sk_c
    return if f.ν < 0 || abs(dot(srvec, sk_c)) >= f.ν * norm(srvec) * norm(sk_c)
        f.matrix = f.matrix + srvec * srvec' / (srvec' * sk_c)
    end
end

function update_hessian_basis!(M, f::ApproxHessianSymmetricRankOne{AllocatingEvaluation}, p)
    update_basis!(f.basis, M, f.p_tmp, p, f.vector_transport_method)
    copyto!(f.p_tmp, p)
    return f.grad_tmp = f.gradient!!(M, f.p_tmp)
end

function update_hessian_basis!(M, f::ApproxHessianSymmetricRankOne{InplaceEvaluation}, p)
    update_basis!(f.basis, M, f.p_tmp, p, f.vector_transport_method)
    copyto!(f.p_tmp, p)
    return f.gradient!!(M, f.grad_tmp, f.p_tmp)
end

@doc """
    ApproxHessianBFGS{E, P, G, T, B<:AbstractBasis{ℝ}, VTR, R<:Real} <: AbstractApproxHessian
A functor to approximate the Hessian by the BFGS update.

# Fields

* `gradient!!` the gradient function (either allocating or mutating, see `evaluation` parameter).
* `scale`
$(_var(:Field, :vector_transport_method))

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
$(_var(:Keyword, :evaluation))
$(_var(:Keyword, :vector_transport_method))
"""
mutable struct ApproxHessianBFGS{
        E, P, G, T, B <: AbstractBasis{ℝ}, VTR <: AbstractVectorTransportMethod,
    } <: AbstractApproxHessian
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
        initial_operator::AbstractMatrix = Matrix{Float64}(
            I, manifold_dimension(M), manifold_dimension(M)
        ),
        basis::B = default_basis(M, typeof(p)),
        scale::Bool = true,
        evaluation = AllocatingEvaluation(),
        vector_transport_method::VTM = default_vector_transport_method(M, typeof(p)),
    ) where {mT <: AbstractManifold, P, G, B <: AbstractBasis{ℝ}, VTM <: AbstractVectorTransportMethod}
    if evaluation == AllocatingEvaluation()
        grad_tmp = gradient(M, p)
    elseif evaluation == InplaceEvaluation()
        grad_tmp = zero_vector(M, p)
        gradient(M, grad_tmp, p)
    end
    return ApproxHessianBFGS{typeof(evaluation), P, G, typeof(grad_tmp), B, VTM}(
        p, gradient, grad_tmp, initial_operator, basis, vector_transport_method, scale
    )
end

function (f::ApproxHessianBFGS{AllocatingEvaluation})(M, p, X)
    # Update Basis if necessary
    if p != f.p_tmp
        update_basis!(f.basis, M, f.p_tmp, p, f.vector_transport_method)
        copyto!(M, f.p_tmp, p)
        f.grad_tmp = f.gradient!!(M, f.p_tmp)
    end

    # Apply Hessian approximation on vector
    return get_vector(
        M, f.p_tmp, f.matrix * get_coordinates(M, f.p_tmp, X, f.basis), f.basis
    )
end

function (f::ApproxHessianBFGS{InplaceEvaluation})(M, Y, p, X)
    # Update Basis if necessary
    if p != f.p_tmp
        update_basis!(f.basis, M, f.p_tmp, p, f.vector_transport_method)
        copyto!(M, f.p_tmp, p)
        f.gradient!!(M, f.grad_tmp, f.p_tmp)
    end

    # Apply Hessian approximation on vector
    Y .= get_vector(M, f.p_tmp, f.matrix * get_coordinates(M, f.p_tmp, X, f.basis), f.basis)

    return Y
end

function update_hessian!(
        M::AbstractManifold, f::ApproxHessianBFGS{AllocatingEvaluation}, p, p_proposal, X
    )
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
    f.matrix =
        f.matrix + yk_c * yk_c' / skyk_c -
        f.matrix * sk_c * sk_c' * f.matrix / dot(sk_c, f.matrix * sk_c)
    return f
end

function update_hessian!(M, f::ApproxHessianBFGS{InplaceEvaluation}, p, p_proposal, X)
    grad_proposal = zero_vector(M, p_proposal)
    f.gradient!!(M, grad_proposal, p_proposal)
    yk_c = get_coordinates(
        M,
        p,
        vector_transport_to(M, p_proposal, grad_proposal, p, f.vector_transport_method) -
            f.grad_tmp,
        f.basis,
    )
    sk_c = get_coordinates(M, p, X, f.basis)
    skyk_c = dot(sk_c, yk_c)
    f.matrix =
        f.matrix + yk_c * yk_c' / skyk_c -
        f.matrix * sk_c * sk_c' * f.matrix / dot(sk_c, f.matrix * sk_c)
    return f
end

function update_hessian_basis!(M, f::ApproxHessianBFGS{AllocatingEvaluation}, p)
    update_basis!(f.basis, M, f.p_tmp, p, f.vector_transport_method)
    copyto!(f.p_tmp, p)
    f.grad_tmp = f.gradient!!(M, f.p_tmp)
    return f
end

function update_hessian_basis!(M, f::ApproxHessianBFGS{InplaceEvaluation}, p)
    update_basis!(f.basis, M, f.p_tmp, p, f.vector_transport_method)
    copyto!(f.p_tmp, p)
    f.gradient!!(M, f.grad_tmp, f.p_tmp)
    return f
end
