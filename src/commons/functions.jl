function maybe_wrap_variable end
"""
    maybe_wrap_variable(v)

For a number variable `v` wrap it in a 0-dimensional array to make it mutable.
Otherwise return the variable as is.
"""
maybe_wrap_variable(v)
maybe_wrap_variable(v::Number) = fill(v)
maybe_wrap_variable(v) = v

function maybe_unwrap_variable end
"""
    maybe_unwrap_variable(p::P, q::P)
    maybe_unwrap_variable(p::P, q::Vector{P})

Undo the wrapping performed by [`maybe_wrap_variable`](@ref).

Given the original input variable `p` and the possibly wrapped variable `q`, return the unwrapped variable,
i.e. if `q` is a 1-element vector of same element-type `P` as the type of `p`,
return this one element.
"""
maybe_unwrap_variable(::P, q) where {P} = q #Default, e.g. also for states: do not unwrap
maybe_unwrap_variable(p::P, q::Vector{P}) where {P} = maybe_unwrap_variable(typeof(p), q)
maybe_unwrap_variable(::Type{P}, q::Vector{P}) where {P} = length(q) == 1 ? q[] : q
maybe_unwrap_variable(::P, q::Array{P, 0}) where {P} = q[]
maybe_unwrap_variable(::Type{P}, q::Array{P, 0}) where {P} = q[]

"""
    MutableManifoldFunction{result, P, F} <: AbstractDecoratedManifoldFunction{F}

A wrapper for a function defined on a manifold to ensure it works on mutable variables,
internally “unwrapping” them to numbers before calling the function that is wrapped.

Since the function works on immutable input types, it is assumed to work allocating.
To use it within objectives of `Manopt.jl`, consider wrapping e.g. gradient or Hessian
functions furthermore in an [`InplaceManifoldFunction`](@ref).

# Fields
* `f::F` : the function to be wrapped of the form `(M, args...) -> v`

The type parameter `result` specifies the type of result. If the result is expected to be a
`:Number`, it is kept as is; for anything else, like a `:Point` or `:TangentVector`, the
result is returned (again) as a mutable variable.

# Constructor

    MutableManifoldFunction(f, p::P, result = :Number)
    MutableManifoldFunction(f, P, result = :Number)

Initialize the wrapper for a function `f` defined on a manifold, where `p` is a point on the manifold,
to store the original point type `P` for the arguments.
"""
struct MutableManifoldFunction{result, P, F} <: AbstractDecoratedManifoldFunction{F}
    f::F
    function MutableManifoldFunction(f::F, ::Type{P}, result::Symbol = :Number) where {F, P}
        return new{result, P, F}(f)
    end
end
# do not “wrap twice”
function MutableManifoldFunction(mmf::MutableManifoldFunction, ::Type, ::Symbol = :Number)
    return mmf
end
# an approximate Hessian already works on the internal (mutable) representation
function MutableManifoldFunction(
        f::AbstractApproximateHessianFunction, ::Type, ::Symbol = :Number
    )
    return f
end
function MutableManifoldFunction(f::F, ::P, result::Symbol = :Number) where {F, P}
    return MutableManifoldFunction(f, P, result)
end
function (f::MutableManifoldFunction{result, P})(M, args...) where {result, P}
    args_unwrapped = map(a -> maybe_unwrap_variable(P, a), args)
    v = f.f(M, args_unwrapped...)
    return result === :Number ? v : maybe_wrap_variable(v)
end
function get_parameter(mmf::MutableManifoldFunction, e::Val, args...)
    return get_parameter(mmf.f, e, args...)
end
function set_parameter!(mmf::MutableManifoldFunction, e::Val, args...)
    return set_parameter!(mmf.f, e, args...)
end
function show(io::IO, mmf::MutableManifoldFunction{result, P}) where {result, P}
    return print(io, "MutableManifoldFunction(", mmf.f, ", ", P, ", :", result, ")")
end
function status_summary(mmf::MutableManifoldFunction; context::Symbol = :default)
    return status_summary(mmf.f; context = context)
end

"""
    InplaceManifoldFunction{result, F} <: AbstractDecoratedManifoldFunction{F}

Wrapper for a function to ensure it works in-place. Since the action to perform to the
provided return value differs per type, the following cases for results are available:

* `:Point` use `copyto!` for a point on a manifold
* `:Points` use an element-wise `copyto!` for points
* `:TangentVector` use `copyto!` for a tangent vector
* `:TangentVectors` use an elementwise `copyto!` for tangent vectors
* `:Number` assume the result to be a 0-dimensional array.
* `:NumberAndTangentVector` for the combination `(c, X)` of a number and a tangent vector – return `c` and handle `X` with the `copyto!` for a tangent vector
* `:MaybeResizeVector` for a vector to return, make sure the size is adapted if needed. This is useful e.g. for return values of sub solvers that might vary in length
* `:Default` (also all other symbols) just use a plain `copyto!`

For those that require an additional point like the tangent vectors, the point is taken as the `point_index` entry of the `args...`

# Fields
* `f::F` : the function to be wrapped of the form `(M, args...) -> v`
* `point_index`: which of the arguments `args...` is the point to be used in the copy
  * the default that works for most functions is `1`
  * for the proximal map `(M, λ, p)` this is index `2`

The type parameter `result` specifies which of the cases above to use.

# Constructor

    InplaceManifoldFunction(f, result = :Point)
"""
struct InplaceManifoldFunction{result, F} <: AbstractDecoratedManifoldFunction{F}
    f::F
    point_index::Int
    function InplaceManifoldFunction(f::F, result::Symbol = :Point; point_index::Int = 1) where {F}
        return new{result, F}(f, point_index)
    end
end
# do not “wrap twice”
function InplaceManifoldFunction(imf::InplaceManifoldFunction, ::Symbol = :Point; point_index::Int = 1)
    return imf
end
function (imf::InplaceManifoldFunction{result})(M, v, args...) where {result}
    (result === :Point) && return copyto!(M, v, imf.f(M, args...))
    (result === :Points) && return map((pa, pb) -> copyto!(M, pa, pb), v, imf.f(M, args...))
    (result === :TangentVector) && return copyto!(M, v, args[imf.point_index], imf.f(M, args...))
    (result === :TangentVectors) && return map((Xa, Xb) -> copyto!(M, Xa, Xb), v, imf.f(M, args...))
    (result === :Number) && return (v[] = imf.f(M, args...))
    # for example (c, X) = costgrad(M, p)
    (result === :NumberAndTangentVector) && return ((c, X) = imf.f(M, args...); copyto!(M, v, X); (c, v))
    # For cases like in ProxBundle where the subsolver can return different sizes, we have to use assign
    if (result === :MaybeResizeVector)
        # For a few in-place assignments, we maybe want to grow/shrink the result vector
        # For example for the prox bundle or convex bundle sub solvers.
        w = imf.f(M, args...)
        (length(v) != length(w)) && resize!(v, length(w))
        return v .= w
    end
    # default: Just copyto! – e.g. for :Vector or :Matrix
    return copyto!(v, imf.f(M, args...))
end
function get_parameter(imf::InplaceManifoldFunction, e::Val, args...)
    return get_parameter(imf.f, e, args...)
end
function set_parameter!(imf::InplaceManifoldFunction, e::Val, args...)
    return set_parameter!(imf.f, e, args...)
end
function show(io::IO, imf::InplaceManifoldFunction{result}) where {result}
    return print(io, "InplaceManifoldFunction(", imf.f, ", :", result, ")")
end
function status_summary(imf::InplaceManifoldFunction; context::Symbol = :default)
    return status_summary(imf.f; context = context)
end

"""
    maybe_wrap_function(f, p, evaluation = InplaceEvaluation(); result = :Number)
    maybe_wrap_function(f, evaluation; result = :Point)

Wrap a function `f` defined on a manifold to work in-place on mutable variables, i.e. first
if the input variable `p` is a number, the function `f` is wrapped in a [`MutableManifoldFunction`](@ref).
If the function then has an [`AllocatingEvaluation`](@ref) as its `evaluation` type, it is wrapped in an [`InplaceManifoldFunction`](@ref) to work in-place of the result.

The first step is skipped if the input variable `p` is not a number, `missing` or not provided.
"""
maybe_wrap_function(f, p, evaluation::AbstractEvaluationType = InplaceEvaluation(); result::Symbol = :Number, point_index = 1) = maybe_wrap_function(f, typeof(p), evaluation; result = result, point_index = point_index)
function maybe_wrap_function(
        f, ::Type{P}, evaluation::AbstractEvaluationType = InplaceEvaluation(); result::Symbol = :Number, point_index = 1
    ) where {P <: Number}
    return maybe_wrap_function(MutableManifoldFunction(f, P, result), evaluation; result = result, point_index = point_index)
end
function maybe_wrap_function(f, ::Type{P}, evaluation::AbstractEvaluationType = InplaceEvaluation(); result::Symbol = :Number, point_index = 1) where {P}
    return maybe_wrap_function(f, evaluation; result = result, point_index = point_index)
end
function maybe_wrap_function(f, ::Missing, evaluation::AbstractEvaluationType = InplaceEvaluation(); result::Symbol = :Number, point_index = 1)
    return maybe_wrap_function(f, evaluation; result = result, point_index = point_index)
end
maybe_wrap_function(f, ::AllocatingEvaluation; result::Symbol = :Point, point_index = 1) = InplaceManifoldFunction(f, result, point_index = point_index)
maybe_wrap_function(f, ::InplaceEvaluation; result::Symbol = :Point, point_index = 1) = f

@doc """
    ApproxHessianFiniteDifference{P, T, G, RTR, VTR, R <: Real} <: AbstractApproximateHessianFunction

A functor to approximate the Hessian by a finite difference of gradient evaluation.

Given a point `p` and a direction `X` and the gradient ``$(_tex(:grad)) f(p)``
of a function ``f`` the Hessian is approximated as follows:
let ``c`` be a stepsize, ``X ∈ $(_math(:TangentSpace))`` a tangent vector and ``q = $_doc_ApproxHessian_step``
be a step in direction ``X`` of length ``c`` following a retraction.
Then the Hessian is approximated by the finite difference of the gradients,
where ``$(_math(:VectorTransport))`` is a vector transport.

$_doc_ApproxHessian_formula

# Fields

* `gradient!`:              the gradient function (either allocating or mutating, see `evaluation` parameter)
* `steplength`:             a step length for the finite difference
$(_fields([:retraction_method, :vector_transport_method]))

## Internal temporary fields

* `grad_tmp`:     a temporary storage for the gradient at the current `p`
* `grad_tmp_dir`: a temporary storage for the gradient at the current `p_dir`
* `p_dir::P`:     a temporary storage for the forward direction (or the ``q`` in the formula)

# Constructor

    ApproxHessianFiniteDifference(M, p, grad_f; kwargs...)

## Keyword arguments

* `steplength=2^-14`: step length ``c`` to approximate the gradient evaluations
$(_kwargs(:evaluation))
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
        tangent_vector = zero_vector(M, maybe_wrap_variable(p)),
        steplength::R = 2^-14,
        retraction_method::RTR = default_retraction_method(M, typeof(p)),
        vector_transport_method::VTR = default_vector_transport_method(M, typeof(p)),
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
    ) where {
        mT <: AbstractManifold, P, G, R <: Real,
        RTR <: AbstractRetractionMethod, VTR <: AbstractVectorTransportMethod,
    }
    p_ = maybe_wrap_variable(p)
    X = copy(M, p_, tangent_vector)
    Y = copy(M, p_, tangent_vector)
    grad_f_ = maybe_wrap_function(grad_f, p, evaluation, result = :TangentVector)
    return ApproxHessianFiniteDifference{typeof(p_), typeof(X), typeof(grad_f_), RTR, VTR, R}(
        p_, grad_f_, X, Y, retraction_method, vector_transport_method, steplength
    )
end
function (f::ApproxHessianFiniteDifference)(M, p, X)
    return f(M, zero_vector(M, p), p, X)
end
function (f::ApproxHessianFiniteDifference)(M, Y, p, X)
    norm_X = norm(M, p, X)
    (norm_X ≈ zero(norm_X)) && return zero_vector!(M, Y, p)
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
$(_fields(:vector_transport_method))

## Internal temporary fields

* `p_tmp`: a temporary storage for the current point `p`.
* `grad_tmp`: a temporary storage for the gradient at the current `p`.
* `matrix`: a temporary storage for the matrix representation of the approximating operator.
* `basis`: a temporary storage for an orthonormal basis at the current `p`.

# Constructor

    ApproxHessianSymmetricRankOne(M, p, gradF; kwargs...)

## Keyword arguments

$(_kwargs(:evaluation))
* `initial_operator=Matrix{Float64}(I, manifold_dimension(M), manifold_dimension(M))`: the matrix representation of the initial approximating operator.
* `basis=`[`default_basis`](@extref `ManifoldsBase.default_basis-Union{Tuple{T}, Tuple{AbstractManifold, Type{T}}} where T`)`(M, typeof(p))`: an orthonormal basis in the tangent space of the initial iterate `p`.
* `nu=-1.0`: the value ``ν`` above; a negative value disables the safeguard on the denominator.
$(_kwargs(:vector_transport_method))
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
        M::mT, p::P, grad_f::G;
        initial_operator::AbstractMatrix = Matrix{Float64}(I, manifold_dimension(M), manifold_dimension(M)),
        basis::B = default_basis(M, typeof(p)),
        nu::R = -1.0,
        vector_transport_method::VTM = default_vector_transport_method(M, typeof(p)),
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
    ) where {
        mT <: AbstractManifold, P, G, B <: AbstractBasis{ℝ}, R <: Real, VTM <: AbstractVectorTransportMethod,
    }
    p_ = maybe_wrap_variable(p)
    X = zero_vector(M, p_)
    grad_f_ = maybe_wrap_function(grad_f, p, evaluation; result = :TangentVector)
    # Fill X with current gradient
    grad_f_(M, X, p_)
    return ApproxHessianSymmetricRankOne{typeof(p_), typeof(grad_f_), typeof(X), B, VTM, R}(
        p_, grad_f_, X, initial_operator, basis, vector_transport_method, nu
    )
end
function (f::ApproxHessianSymmetricRankOne)(M, p, X)
    return f(M, zero_vector(M, p), p, X)
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
"""
    update_hessian_basis!(M, f, p)

Update the basis of tangent vectors and the stored gradient of the approximate Hessian `f`
when moving to the point `p`, using the vector transport stored in `f`.
"""
function update_hessian_basis!(M, f::ApproxHessianSymmetricRankOne, p)
    update_basis!(f.basis, M, f.p_tmp, p, f.vector_transport_method)
    copyto!(f.p_tmp, p)
    return f.gradient!(M, f.grad_tmp, f.p_tmp)
end

@doc """
    ApproxHessianBFGS{P, G, T, B<:AbstractBasis{ℝ}, VTR} <: AbstractApproximateHessianFunction

A functor to approximate the Hessian by the BFGS update.

# Fields

* `gradient!`: the gradient function (either allocating or mutating, see `evaluation` parameter).
* `scale::Bool`: a flag stored for a scaling of the initial approximating operator; it is currently not used in the update.
$(_fields(:vector_transport_method))

## Internal temporary fields

* `p_tmp`: a temporary storage for the current point `p`.
* `grad_tmp`: a temporary storage for the gradient at the current `p`.
* `matrix`: a temporary storage for the matrix representation of the approximating operator.
* `basis`: a temporary storage for an orthonormal basis at the current `p`.

# Constructor
    ApproxHessianBFGS(M, p, gradF; kwargs...)

## Keyword arguments

$(_kwargs(:evaluation))
* `initial_operator=Matrix{Float64}(I, manifold_dimension(M), manifold_dimension(M))`: the matrix representation of the initial approximating operator.
* `basis=`[`default_basis`](@extref `ManifoldsBase.default_basis-Union{Tuple{T}, Tuple{AbstractManifold, Type{T}}} where T`)`(M, typeof(p))`: an orthonormal basis in the tangent space of the initial iterate `p`.
* `scale=true`: the value to store in the `scale` field above.
$(_kwargs(:vector_transport_method))
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
        M::mT, p::P, grad_f::G;
        initial_operator::AbstractMatrix = Matrix{Float64}(
            I, manifold_dimension(M), manifold_dimension(M)
        ),
        basis::B = default_basis(M, typeof(p)),
        scale::Bool = true,
        vector_transport_method::VTM = default_vector_transport_method(M, typeof(p)),
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
    ) where {mT <: AbstractManifold, P, G, B <: AbstractBasis{ℝ}, VTM <: AbstractVectorTransportMethod}
    p_ = maybe_wrap_variable(p)
    X = zero_vector(M, p_)
    grad_f_ = maybe_wrap_function(grad_f, p, evaluation; result = :TangentVector)
    grad_f_(M, X, p)
    return ApproxHessianBFGS{typeof(p_), typeof(grad_f_), typeof(X), B, VTM}(
        p_, grad_f_, X, initial_operator, basis, vector_transport_method, scale
    )
end
function (f::ApproxHessianBFGS)(M, p, X)
    return f(M, zero_vector(M, p), p, X)
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

_doc_reflect_prox = """
    reflect(M, pr::Function, x; kwargs...)
    reflect!(M, q, pr::Function, x; kwargs...)

Reflect the point `x` from the manifold `M` at the point ``p = $(_tex(:prox))(x)``,
where the proximal map is given by `pr`.

The formula is given by

```math
$(_tex(:reflect))_p(x) = $(_tex(:retr))_p(-$(_tex(:invretr))_p x),
```
where ``$(_tex(:retr))`` and ``$(_tex(:invretr))`` denote a retraction and an inverse retraction, respectively.

This can also be done in place of `q`.

## Keyword Arguments

$(_kwargs([:retraction_method, :inverse_retraction_method]))
* `X=zero_vector(M,p)`: temporary memory `reflect!` uses to compute the inverse retraction in place;
  the allocating `reflect` ignores this keyword.
"""
@doc "$(_doc_reflect_prox)"
reflect(M::AbstractManifold, pr::Function, x; kwargs...) = reflect(M, pr(x), x; kwargs...)
@doc "$(_doc_reflect_prox)"
function reflect!(M::AbstractManifold, q, pr::Function, x; kwargs...)
    return reflect!(M, q, pr(x), x; kwargs...)
end

_doc_reflect = """
    reflect(M, p, x; kwargs...)
    reflect!(M, q, p, x; kwargs...)

Reflect the point `x` from the manifold `M` at point `p`.

The formula is given by

```math
$(_tex(:reflect))_p(x) = $(_tex(:retr))_p(-$(_tex(:invretr))_p x),
```

where ``$(_tex(:retr))`` and ``$(_tex(:invretr))`` denote a retraction and an inverse
retraction, respectively.
This can also be done in place of `q`.

## Keyword Arguments

$(_kwargs([:retraction_method, :inverse_retraction_method]))
$(_kwargs(:X))
  used by `reflect!` as temporary memory to compute the inverse retraction in place;
  the allocating `reflect` ignores this keyword.
"""
@doc "$(_doc_reflect)"
function reflect(
        M::AbstractManifold, p, x;
        retraction_method = default_retraction_method(M, typeof(p)),
        inverse_retraction_method = default_inverse_retraction_method(M, typeof(p)),
        X = nothing,
    )
    return retract(
        M, p, -inverse_retract(M, p, x, inverse_retraction_method), retraction_method
    )
end
@doc "$(_doc_reflect)"
function reflect!(
        M::AbstractManifold, q, p, x;
        retraction_method = default_retraction_method(M, typeof(p)),
        inverse_retraction_method = default_inverse_retraction_method(M, typeof(p)),
        X = zero_vector(M, p),
    )
    inverse_retract!(M, X, p, x, inverse_retraction_method)
    X .*= -1
    return retract!(M, q, p, X, retraction_method)
end
