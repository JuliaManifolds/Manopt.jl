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
