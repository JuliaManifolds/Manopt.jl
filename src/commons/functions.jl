"""
    maybe_wrap_function(f, p, evaluation = AllocatingEvaluation(); result = :TangentVector)

Wrap a function `f` defined on a manifold `M` to work
* in-place, that is copy over the result if it is not yet in-place
* on mutating input `p`, i.e. use a 1-element vector to store numbers.
"""

function maybe_wrap_variable end
"""
    maybe_wrap_variable(v)

For a number variable `v` wrap it in an 1-element vector to make it mutable.
Otherwise return the variable as is.
"""
maybe_wrap_variable(v)
maybe_wrap_variable(v::Number) = [v]
maybe_wrap_variable(v) = v

function maybe_unwrap_variable end
"""
    maybe_unwrap_variable(p::P, q::P)
    maybe_unwrap_variable(p::P, q::Vector{P})

Undo the wrapping performed by `maybe_wrap_variable`, i.e. given the original
input variable `p` and the possibly wrapped variable `q`, return the unwrapped variable,
i.e. if `q` is a 1-element vector of same element-type `P` as the type of `p`,
return this one element.
"""
maybe_unwrap_variable(::P, q::P) where {P} = q
maybe_unwrap_variable(p::P, q::Vector{P}) where {P} = maybe_unwrap_variable(typeof(p), q)
maybe_unwrap_variable(::Type{P}, q::Vector{P}) where {P} = length(q) == 1 ? q[] : q

"""
    MutableManifoldFunction{P, F}

A wrapper for a function defined on a manifold to ensure it works on mutable variables,
internally “unwrapping” them to numbers before calling the function that is wrapped.

Since the function is working on immutable input types, it is assumed to work allocating,
i.e. to be used within objectives of `Manopt.jl`, consider wrapping e.g. gradient or Hessian
functions furthermore in a [`InplaceManifoldFunction`](@ref).

## Fields
* `f::F` : the function to be wrapped of the form `(M, args...) -> v`
* `result::Symbol`: specify the type of result. If the result is expected to be a `:Number`,
  it is kept as is, for anything else, like a `:Point` or `:TangentVector`, the result is
  returned (again) as a mutable variable

## Constructor

    MutableManifoldFunction(f, p::P, result = :Number)
    MutableManifoldFunction(f, P, result = :Number)

Initialise the wrapper for a function `f` defined on a manifold, where `p` is a point on the manifold,
to store the original point type `P` for the Arguments.
"""
struct MutableManifoldFunction{P, F}
    f::F
    result::Symbol
    function MutableManifoldFunction(f::F, ::Type{P}, result::Symbol = :Number) where {F, P}
        return new{P, F}(f, result)
    end
end
function MutableManifoldFunction(f::F, ::P, result::Symbol = :Number) where {F, P}
    return MutableManifoldFunction(f, P; result = result)
end
function (f::MutableManifoldFunction{P})(M, args...) where {P}
    args_unwrapped = map(a -> maybe_unwrap_variable(P, a), args)
    v = f.f(M, args_unwrapped...)
    return f.result === :Number ? v : maybe_wrap_variable(v)
end

"""
    InplaceManifoldFunction{F}

Wrapper for a function to ensure it works in-place.

# Fields
* `f::F` : the function to be wrapped of the form `(M, args...) -> v`
* `result::Symbol`: specify the type
  * `:Point` uses the corresponding `copyto!` for points
  * `:TangentVector` uses the corresponding `copyto!` for tangentvectors
    this type assumes, that the first argument is the point `p` the tangent vector is at.
  * `:Number` the result is a number. We hence assume that `v` is a zero- or one-dimensional array and store the value using `v[]`.

# Constructor

    InplaceManifoldFunction(f, result = :Point)
"""
struct InplaceManifoldFunction{F}
    f::F
    result::Symbol
    function InplaceManifoldFunction(f::F, result::Symbol = :Point) where {F}
        return new{F}(f, result)
    end
end
function (f!::InplaceManifoldFunction)(M, v, p, args...)
    (f!.result === :Point) && return copyto!(M, v, f!.f(M, p, args...))
    (f!.result === :TangentVector) && return copyto!(M, v, p, f!.f(M, p, args...))
    (f!.result === :Number) && return (v[] = f!.f(M, p, args...))
    # default: Just copyto!
    return copyto!(v, f!.f(M, args...))
end

"""
    maybe_wrap_function(f, p, evaluation = InplaceEvaluation(); result = :Number)
    maybe_wrap_function(f, evaluation = InplaceEvaluation(); result = :Number)

Wrap a function `f` defined on a manifold to work in-place on mutable variables, i.e. first
if the input variable `p` is a number, the function `f` is wrapped in a [`MutableManifoldFunction`](@ref).
If then the function has an [`AllocatingManifoldFunction`](@ref) as `evaluation` type, it is wrapped in a [`InplaceManifoldFunction`](@ref) to work in-place of the result.

The first step is skipped if the input variable `p` is not a number, `missing` or not provided.
"""
maybe_wrap_function(f, p, evaluation::AbstractEvaluationType = InplaceEvaluation(); result::Symbol = :Number) = maybe_wrap_function(f, typeof(p), evaluation; result = result)
function maybe_wrap_function(
        f, ::Type{P}, evaluation::AbstractEvaluationType = InplaceEvaluation(); result::Symbol = :Number
    ) where {P <: Number}
    return maybe_wrap_function(MutableManifoldFunction(f, P, result), evaluation; result = result)
end
function maybe_wrap_function(f, ::Type{P}, evaluation::AbstractEvaluationType = InplaceEvaluation(); result::Symbol = :Number) where {P}
    return maybe_wrap_function(f, evaluation; result = result)
end
function maybe_wrap_function(f, ::Missing, evaluation::AbstractEvaluationType = InplaceEvaluation(); result::Symbol = :Number)
    return maybe_wrap_function(f, evaluation; result = result)
end
maybe_wrap_function(f, ::AllocatingEvaluation; result::Symbol = :Point) = InplaceManifoldFunction(f, result)
maybe_wrap_function(f, ::InplaceEvaluation; result::Symbol = :Point) = f

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
        tangent_vector = zero_vector(M, p),
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

$(_kwargs(:evaluation))
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

$(_kwargs(:evaluation))
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
