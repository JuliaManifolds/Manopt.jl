#
# Manopt.jl – basic algorithms
#
# A collection of simple algorithms that might be helpful
#
# ---
# Manopt.jl – Ronny Bergmann – 2017-07-06
import Statistics: mean, median
export mean, median, variance
export useGradientDescent, useSubgradientDescent, useProximalPoint, useCyclicProximalPoint, useDouglasRachford
# Indicators for Algorithms
struct useGradientDescent end
struct useSubgradientDescent end
struct useProximalPoint end
struct useCyclicProximalPoint end
struct useDouglasRachford end

"""
    y = mean(M,x)

compute the Riemannian center of mass of the data given by the vector of
[`MPoint`](@ref)s `x` on the [`Manifold`](@ref) `M`.
calculates the Riemannian Center of Mass (Karcher mean) of the input data `x`
as an `Array` of [`MPoint`](@ref)s on the [`Manifold`](@ref) with a
[`steepestDescent`](@ref) or a [`cyclicProximalPoint`](@ref).

# Optional
* `initialValue` – (`x[1]`) the value to initialize the algorithm to
* `method` – (`:GradientDescent`) symbol indicating the algorithm to use,
  so the second variant is `:CyclicProximalPoint`
* `weights` – (`1/n`) compute a weighted Karcher mean, i.e. the dault is to
  set all weights to be `1/n` where `n` is the length of `x`.

as well as optional parameters that are passed down to the corresponding algorithm
"""
function mean(M::mT, x::Vector{T};
    initialValue::T = x[1],
    method::Symbol = :GradientDescent,
    weights = 1/length(x)*ones(length(x)),
    kwargs...)::T where {mT <: Manifold, T <: MPoint}
  return mean_(M,x,initialValue,weights,Val(method);kwargs...)
end
function mean_(M,x,x0,w,::Val{:GradientDescent};kwargs...)
    F = y -> sum(w .* 1/2 .* distance.(Ref(M),Ref(y),x).^2)
    ∇F = y -> sum( w.*gradDistance.(Ref(M),x,Ref(y)))
    return steepestDescent(M,F,∇F,x0; kwargs...)
end
function mean_(M,x,x0,w,::Val{:CyclicProximalPoint};kwargs...)
    F = y -> sum(w .* 1/2 .* distance.(Ref(M),Ref(y),x).^2)
    proxes = Function[ (λ,y) -> proxDistance(M, λ*wi, xi, y) for (wi,xi) in zip(w,x) ]
    return cyclicProximalPoint(M,F,proxes,x0; kwargs...)
end
@doc doc"""
    y = median(M,x)

compute the median of the data given by the vector of
[`MPoint`](@ref)s `x` on the [`Manifold`](@ref) `M`.
calculates the Riemannian Center of Mass (Karcher mean) of the input data `x`
as an `Array` of [`MPoint`](@ref)s on the [`Manifold`](@ref) with a
[`cyclicProximalPoint`](@ref).

# Optional

* `initialValue` – (`x[1]`) the value to initialize the algorithm to
* `method` – (`:CyclicProximalPoint`) symbol indicating the algorithm to use
* `weights` – (`1/n`) compute a weighted Karcher mean, i.e. the dault is to
  set all weights to be `1/n` where `n` is the length of `x`.

as well as optional parameters that are passed down to the corresponding algorithm
"""
function median(M::mT, x::Vector{T};
    initialValue::T = x[1],
    method::Symbol = :CyclicProximalPoint,
    weights = 1/length(x)*ones(length(x)),
    kwargs...)::T where {mT <: Manifold, T <: MPoint}
  return median_(M,x,initialValue,weights,Val(method);kwargs...)
end
function median_(M,x,x0,w,::Val{:CyclicProximalPoint};kwargs...)
    F = y -> sum(w .* 1/2 .* distance.(Ref(M),Ref(y),x))
    proxes = Function[ (λ,y) -> proxDistance(M, λ*wi, xi, y,1) for (wi,xi) in zip(w,x) ]
    return cyclicProximalPoint(M,F,proxes,x0; kwargs...)
end

"""
    variance(M,x)

returns the variance of the vector `x` of [`MPoint`](@ref)s on the
[`Manifold`](@ref) `M`
"""
function variance(M::mT,x::Vector{T};kwargs...) where {mT<:Manifold,T<:MPoint}
  meanX = mean(M,x;kwargs...)
  return 1/( (length(x)-1)*manifoldDimension(M) ) * sum( distance.(Ref(M),Ref(meanX),x).^2 )
end
