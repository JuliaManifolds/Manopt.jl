"""
    AllocatingManifoldFunction{F}

Wrapper for a function that does not work in-place but allocates, i.e. a function of the form
`f(M, args...) = v` is wrapped herein to work as an in-place variant
`f!(M, v, args...) = v` to be used within Manopt

# Fields
* `f::F` : the function to be wrapped of the form `(M, args...) -> v`
* `result::Symbol`: specify the type
  * `:Point` uses the corresponding `copyto!` for points
  * `:TangentVector` uses the corresponding `copyto!` for tangentvectors
    this type assumes, that the first argument is the point `p` the tangent vector is at.
  * `:Number` the result is a number and hence can not use `copyto!`, we hence assume `v`
    is a 0-dimensional array.

  all other symbols use a call of `copyto!`


# Constructor

    AllocatingManifoldFunction(f, result = :Point)
"""
struct AllocatingManifoldFunction{F}
    f::F
    result::Symbol
    function AllocatingManifoldFunction(f::F, result::Symbol = :Point) where {F}
        return new{F}(f, result)
    end
end
function (f!::AllocatingManifoldFunction)(M, v, args...)
    (f!.result === :Point) && return copyto!(M, v, f!.f(M, args...))
    (f!.result === :TangentVector) && return copyto!(M, v, first(args...), f!.f(M, args...))
    (f!.result === :Number) && return (v[] = f!.f(M, args...))
    # default: Just copyto!
    return copyto!(v, f!.f(M, args...))
end

# TODO: Maybe also re-add the evaluation keyword here again
@doc """
    ApproxHessianFiniteDifference{P, T, G, RTR, VTR, R <: Real} <: AbstractApproximateHessianFunction

A functor to approximate the Hessian by a finite difference of gradient evaluation.

Given a point `p` and a direction `X` and the gradient ``$(_tex(:grad)) f(p)``
of a function ``f`` the Hessian is approximated as follows:
let ``c`` be a stepsize, ``X ∈ $(_math(:TangentSpace))`` a tangent vector and ``q = $_doc_ApproxHessian_step``
be a step in direction ``X`` of length ``c`` following a retraction
Then the Hessian is approximated by the finite difference of the gradients,
where ``$(_math(:VectorTransport))`` is a vector transport.

$_doc_ApproxHessian_formula

 # Fields

* `gradient!`:              the gradient function (either allocating or mutating, see `evaluation` parameter)
* `step_length`:             a step length for the finite difference
$(_kwargs([:retraction_method, :vector_transport_method]))

## Internal temporary fields

* `grad_tmp`:     a temporary storage for the gradient at the current `p`
* `grad_dir_tmp`: a temporary storage for the gradient at the current `p_dir`
* `p_dir::P`:     a temporary storage to the forward direction (or the ``q`` in the formula)

# Constructor

    ApproximateFiniteDifference(M, p, grad_f; kwargs...)

## Keyword arguments

* `steplength=2^{-14}`: step length ``c`` to approximate the gradient evaluations
$(_kwargs([:retraction_method, :vector_transport_method]))
"""
mutable struct ApproxHessianFiniteDifference{P, T, G, RTR, VTR, R <: Real} <: AbstractApproximateHessianFunction
    p_dir::P
    gradient!::G
    grad_tmp::T
    grad_tmp_dir::T
    retraction_method::RTR
    vector_transport_method::VTR
    steplength::R
end
function ApproxHessianFiniteDifference(
        M::mT, p::P, grad_f::G;
        tangent_vector = zero_vector(M, p),
        steplength::R = 2^-14,
        retraction_method::RTR = default_retraction_method(M, typeof(p)),
        vector_transport_method::VTR = default_vector_transport_method(M, typeof(p)),
    ) where {
        mT <: AbstractManifold, P, G, R <: Real,
        RTR <: AbstractRetractionMethod, VTR <: AbstractVectorTransportMethod,
    }
    X = copy(M, p, tangent_vector)
    Y = copy(M, p, tangent_vector)
    return ApproxHessianFiniteDifference{P, typeof(X), G, RTR, VTR, R}(
        p, grad_f, X, Y, retraction_method, vector_transport_method, steplength
    )
end
function (f::ApproxHessianFiniteDifference)(M, Y, p, X)
    norm_X = norm(M, p, X)
    (norm_X ≈ zero(norm_X)) && return zero_vector!(M, X, p)
    c = f.steplength / norm_X
    f.gradient!(M, f.grad_tmp, p)
    retract!(M, f.p_dir, p, c * X, f.retraction_method)
    f.gradient!(M, f.grad_tmp_dir, f.p_dir)
    vector_transport_to!(
        M, f.grad_tmp_dir, f.p_dir, f.grad_tmp_dir, p, f.vector_transport_method
    )
    Y .= (1 / c) .* (f.grad_tmp_dir .- f.grad_tmp)
    return Y
end

@doc """
    ApproxHessianSymmetricRankOne{P, G, T, B<:AbstractBasis{ℝ}, VTR, R<:Real} <: AbstractApproximateHessianFunction

A functor to approximate the Hessian by the symmetric rank one update.

# Fields

* `gradient!`: the gradient function (either allocating or mutating, see `evaluation` parameter).
* `ν`: a small real number to ensure that the denominator in the update does not become too small and thus the method does not break down.
$(_kwargs(:vector_transport_method)).

## Internal temporary fields

* `p_tmp`: a temporary storage the current point `p`.
* `grad_tmp`: a temporary storage for the gradient at the current `p`.
* `matrix`: a temporary storage for the matrix representation of the approximating operator.
* `basis`: a temporary storage for an orthonormal basis at the current `p`.

# Constructor

    ApproxHessianSymmetricRankOne(M, p, gradF; kwargs...)

## Keyword arguments

* `initial_operator=Matrix{Float64}(I, manifold_dimension(M), manifold_dimension(M))`) the matrix representation of the initial approximating operator.
* `basis=`[`DefaultOrthonormalBasis`](@extref `ManifoldsBase.DefaultOrthonormalBasis`) an orthonormal basis in the tangent space of the initial iterate p.
* `nu` (`-1`)
"""
mutable struct ApproxHessianSymmetricRankOne{P, G, T, B <: AbstractBasis{ℝ}, VTR, R <: Real} <: AbstractApproximateHessianFunction
    p_tmp::P
    gradient!::G
    grad_tmp::T
    matrix::Matrix
    basis::B
    vector_transport_method::VTR
    ν::R
end
function ApproxHessianSymmetricRankOne(
        M::mT, p::P, gradient::G;
        initial_operator::AbstractMatrix = Matrix{Float64}(I, manifold_dimension(M), manifold_dimension(M)),
        basis::B = default_basis(M, typeof(p)),
        nu::R = -1.0,
        vector_transport_method::VTM = default_vector_transport_method(M, typeof(p)),
    ) where {
        mT <: AbstractManifold, P, G, B <: AbstractBasis{ℝ}, R <: Real, VTM <: AbstractVectorTransportMethod,
    }
    grad_tmp = zero_vector(M, p)
    gradient(M, grad_tmp, p)
    return ApproxHessianSymmetricRankOne{P, G, typeof(grad_tmp), B, VTM, R}(
        p, gradient, grad_tmp, initial_operator, basis, vector_transport_method, nu
    )
end
function (f::ApproxHessianSymmetricRankOne)(M, Y, p, X)
    # Update Basis if necessary
    if p != f.p_tmp
        update_basis!(f.basis, M, f.p_tmp, p, f.vector_transport_method)
        copyto!(f.p_tmp, p)
        f.gradient!(M, f.grad_tmp, f.p_tmp)
    end
    # Apply Hessian approximation on vector
    Y .= get_vector(M, f.p_tmp, f.matrix * get_coordinates(M, f.p_tmp, X, f.basis), f.basis)
    return Y
end
function update_hessian!(M::AbstractManifold, f::ApproxHessianSymmetricRankOne, p, p_proposal, X)
    grad_proposal = zero_vector(M, p_proposal)
    f.gradient!(M, grad_proposal, p_proposal)
    yk_c = get_coordinates(
        M, p,
        vector_transport_to(M, p_proposal, grad_proposal, p, f.vector_transport_method) - f.grad_tmp,
        f.basis,
    )
    sk_c = get_coordinates(M, p, X, f.basis)
    srvec = yk_c - f.matrix * sk_c
    return if f.ν < 0 || abs(dot(srvec, sk_c)) >= f.ν * norm(srvec) * norm(sk_c)
        f.matrix = f.matrix + srvec * srvec' / (srvec' * sk_c)
    end
end
function update_hessian_basis!(M, f::ApproxHessianSymmetricRankOne, p)
    update_basis!(f.basis, M, f.p_tmp, p, f.vector_transport_method)
    copyto!(f.p_tmp, p)
    return f.gradient!(M, f.grad_tmp, f.p_tmp)
end

@doc """
    ApproxHessianBFGS{P, G, T, B<:AbstractBasis{ℝ}, VTR, R<:Real} <: AbstractApproximateHessianFunction

A functor to approximate the Hessian by the BFGS update.

# Fields

* `gradient!` the gradient function (either allocating or mutating, see `evaluation` parameter).
* `scale`
$(_fields(:vector_transport_method))

## Internal temporary fields

* `p_tmp` a temporary storage the current point `p`.
* `grad_tmp` a temporary storage for the gradient at the current `p`.
* `matrix` a temporary storage for the matrix representation of the approximating operator.
* `basis` a temporary storage for an orthonormal basis at the current `p`.

# Constructor
    ApproxHessianBFGS(M, p, gradF; kwargs...)

## Keyword arguments

* `initial_operator` (`Matrix{Float64}(I, manifold_dimension(M), manifold_dimension(M))`) the matrix representation of the initial approximating operator.
* `basis=`[`DefaultOrthonormalBasis`](@extref `ManifoldsBase.DefaultOrthonormalBasis`)) an orthonormal basis in the tangent space of the initial iterate p.
* `nu` (`-1`)
"""
mutable struct ApproxHessianBFGS{
        P, G, T, B <: AbstractBasis{ℝ}, VTR <: AbstractVectorTransportMethod,
    } <: AbstractApproximateHessianFunction
    p_tmp::P
    gradient!::G
    grad_tmp::T
    matrix::Matrix
    basis::B
    vector_transport_method::VTR
    scale::Bool
end
function ApproxHessianBFGS(
        M::mT, p::P, gradient::G;
        initial_operator::AbstractMatrix = Matrix{Float64}(
            I, manifold_dimension(M), manifold_dimension(M)
        ),
        basis::B = default_basis(M, typeof(p)),
        scale::Bool = true,
        vector_transport_method::VTM = default_vector_transport_method(M, typeof(p)),
    ) where {mT <: AbstractManifold, P, G, B <: AbstractBasis{ℝ}, VTM <: AbstractVectorTransportMethod}
    grad_tmp = zero_vector(M, p)
    gradient(M, grad_tmp, p)
    return ApproxHessianBFGS{P, G, typeof(grad_tmp), B, VTM}(
        p, gradient, grad_tmp, initial_operator, basis, vector_transport_method, scale
    )
end
function (f::ApproxHessianBFGS)(M, Y, p, X)
    # Update Basis if necessary
    if p != f.p_tmp
        update_basis!(f.basis, M, f.p_tmp, p, f.vector_transport_method)
        copyto!(M, f.p_tmp, p)
        f.gradient!(M, f.grad_tmp, f.p_tmp)
    end
    # Apply Hessian approximation on vector
    Y .= get_vector(M, f.p_tmp, f.matrix * get_coordinates(M, f.p_tmp, X, f.basis), f.basis)
    return Y
end
function update_hessian!(M, f::ApproxHessianBFGS, p, p_proposal, X)
    grad_proposal = zero_vector(M, p_proposal)
    f.gradient!(M, grad_proposal, p_proposal)
    yk_c = get_coordinates(
        M, p,
        vector_transport_to(M, p_proposal, grad_proposal, p, f.vector_transport_method) - f.grad_tmp,
        f.basis,
    )
    sk_c = get_coordinates(M, p, X, f.basis)
    skyk_c = dot(sk_c, yk_c)
    f.matrix =
        f.matrix + yk_c * yk_c' / skyk_c -
        f.matrix * sk_c * sk_c' * f.matrix / dot(sk_c, f.matrix * sk_c)
    return f
end

function update_hessian_basis!(M, f::ApproxHessianBFGS, p)
    update_basis!(f.basis, M, f.p_tmp, p, f.vector_transport_method)
    copyto!(f.p_tmp, p)
    f.gradient!(M, f.grad_tmp, f.p_tmp)
    return f
end

@doc """
    LagrangianCost{CO,T} <: AbstractConstrainedFunction{T}

Implement the Lagrangian of a [`ConstrainedManifoldObjective`](@ref) `co`.

```math
$(_tex(:Cal, "L"))(p; μ, λ) = f(p) + $(_tex(:sum, "i=1", "m")) μ_ig_i(p) + $(_tex(:sum, "j=1", "n")) λ_jh_j(p)
```

# Fields

* `co::CO`, `μ::T`, `λ::T` as mentioned, where `T` represents a vector type.

# Constructor

    LagrangianCost(co, μ, λ)

Create a functor for the Lagrangian with fixed dual variables.

# Example

When you directly want to evaluate the Lagrangian ``$(_tex(:Cal, "L"))``
you can also call

```
LagrangianCost(co, μ, λ)(M,p)
```
"""
mutable struct LagrangianCost{CO, T} <: AbstractConstrainedFunction{T}
    co::CO
    μ::T
    λ::T
end
function (lc::LagrangianCost)(M, p)
    c = get_cost(M, lc.co, p)
    g = get_inequality_constraint(M, lc.co, p, :)
    h = get_equality_constraint(M, lc.co, p, :)
    (length(g) > 0) && (c += sum(lc.μ .* g))
    (length(h) > 0) && (c += sum(lc.λ .* h))
    return c
end
function show(io::IO, lc::LagrangianCost)
    return print(io, "LagrangianCost\n$(_MANOPT_INDENT)with μ=$(lc.μ), λ=$(lc.λ)")
end

@doc """
    LagrangianGradient{CO,T}

The gradient of the Lagrangian of a [`ConstrainedManifoldObjective`](@ref) `co`
with respect to the variable ``p``. The formula reads

```math
$(_tex(:grad))_p $(_tex(:Cal, "L"))(p; μ, λ)
= $(_tex(:grad)) f(p) + $(_tex(:sum, "i=1", "m")) μ_i $(_tex(:grad)) g_i(p) + $(_tex(:sum, "j=1", "n")) λ_j $(_tex(:grad)) h_j(p)
```

# Fields

* `co::CO`, `μ::T`, `λ::T` as mentioned, where `T` represents a vector type.

# Constructor

    LagrangianGradient(co, μ, λ)

Create a functor for the Lagrangian with fixed dual variables.

# Example

When you directly want to evaluate the gradient of the Lagrangian ``$(_tex(:grad))_p $(_tex(:Cal, "L"))``
you can also call `LagrangianGradient(co, μ, λ)(M,p)` or `LagrangianGradient(co, μ, λ)(M,X,p)` for the in-place variant.
"""
mutable struct LagrangianGradient{CO, T} <: AbstractConstrainedFunction{T}
    co::CO
    μ::T
    λ::T
end
function (lg::LagrangianGradient)(M, p)
    X = zero_vector(M, p)
    return lg(M, X, p)
end
function (lg::LagrangianGradient)(M, X, p)
    Y = copy(M, p, X)
    get_gradient!(M, X, lg.co, p)
    m = inequality_constraints_length(lg.co)
    n = equality_constraints_length(lg.co)
    for i in 1:m
        get_grad_inequality_constraint!(M, Y, lg.co, p, i)
        copyto!(M, X, p, X + lg.μ[i] * Y)
    end
    for j in 1:n
        get_grad_equality_constraint!(M, Y, lg.co, p, j)
        copyto!(M, X, p, X + lg.λ[j] * Y)
    end
    return X
end
function show(io::IO, lg::LagrangianGradient)
    return print(io, "LagrangianGradient\n$(_MANOPT_INDENT)with μ=$(lg.μ), λ=$(lg.λ)")
end

@doc """
    LagrangianHessian{CO, V, T}

The Hessian of the Lagrangian of a [`ConstrainedManifoldObjective`](@ref) `co`
with respect to the variable ``p``. The formula reads

```math
$(_tex(:Hess))_p $(_tex(:Cal, "L"))(p; μ, λ)[X]
= $(_tex(:Hess)) f(p) + $(_tex(:sum, "i=1", "m")) μ_i $(_tex(:Hess)) g_i(p)[X] + $(_tex(:sum, "j=1", "n")) λ_j $(_tex(:Hess)) h_j(p)[X]
```

# Fields

* `co::CO`, `μ::T`, `λ::T` as mentioned, where `T` represents a vector type.

# Constructor

    LagrangianHessian(co, μ, λ)

Create a functor for the Lagrangian with fixed dual variables.

# Example

When you directly want to evaluate the Hessian of the Lagrangian ``$(_tex(:Hess))_p $(_tex(:Cal, "L"))``
you can also call `LagrangianHessian(co, μ, λ)(M, p, X)` or `LagrangianHessian(co, μ, λ)(M, Y, p, X)` for the in-place variant.
"""
mutable struct LagrangianHessian{CO, T} <: AbstractConstrainedFunction{T}
    co::CO
    μ::T
    λ::T
end
function (lH::LagrangianHessian)(M, p, X)
    Y = zero_vector(M, p)
    return lH(M, Y, p, X)
end
function (lH::LagrangianHessian)(M, Y, p, X)
    Z = copy(M, p, X)
    get_hessian!(M, Y, lH.co, p, X)
    n = inequality_constraints_length(lH.co)
    m = equality_constraints_length(lH.co)
    for i in 1:n
        get_hess_inequality_constraint!(M, Z, lH.co, p, X, i)
        copyto!(M, Y, p, Y + lH.μ[i] * Z)
    end
    for j in 1:m
        get_hess_equality_constraint!(M, Z, lH.co, p, X, j)
        copyto!(M, Y, p, Y + lH.λ[j] * Z)
    end
    return Y
end
function show(io::IO, lh::LagrangianHessian)
    return print(io, "LagrangianHessian\n$(_MANOPT_INDENT)with μ=$(lh.μ), λ=$(lh.λ)")
end
