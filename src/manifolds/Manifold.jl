#
#      Manifold -- a manifold defined via its data types:
#  * A point on the manifold, MPoint
#  * A point in an tangential space MTVector
#
import Base.LinAlg: norm, dot
import Base: exp, log, mean, median, +, -, *, /, ==, show
# introcude new types
export MPoint, MTVector
# introduce new functions
export distance, exp, log, norm, dot, manifoldDimension
export mean, median, variance, geodesic, midPoint, addNoise
# introcude new algorithms
export proxTV, proxDistanceSquared, proxTVSquared
"""
    Manifold - an abstract Manifold to keep global information on a specific manifold
"""
abstract type Manifold end

"""
    MPoint - an abstract point on a Manifold
"""
abstract type MPoint end

"""
      MTVector - a point on a tangent plane of a base point, which might
  be null if the tangent space is fixed/known to spare memory.
"""
abstract type MTVector end

# scale tangential vectors
*{T <: MTVector}(ξ::T,s::Number)::T = T(s*ξ.value,ξ.base)
*{T <: MTVector}(s::Number, ξ::T) = T(s*ξ.value,ξ.base)
*{T <: MTVector}(ξ::Vector{T},s::Number) = s*ones(length(ξ))*ξ
*{T <: MTVector}(s::Number, ξ::Vector{T}) = s*ones(length(ξ))*ξ
# /
/{T <: MTVector}(ξ::T,s::Number) = T(s/ξ.value,ξ.base)
/{T <: MTVector}(s::Number, ξ::T) = T(s/ξ.value,ξ.base)
/{T <: MTVector}(ξ::Vector{T},s::Number) = s*ones(length(ξ))/ξ
/{T <: MTVector}(s::Number, ξ::Vector{T}) = s*ones(length(ξ))/ξ
# + - of MTVectors
function +{T <: MTVector}(ξ::T,ν::T)::T
  if sameBase(ξ,ν)
    return T(ξ.value+ν.value,ξ.base)
  else
    throw(ErrorException("Can't add two tangential vectors belonging to
      different tangential spaces."))
  end
end
function -{T <: MTVector}(ξ::T,ν::T)::T
  if sameBase(ξ,ν)
    return T(ξ.value-ν.value,ξ.base)
  else
    throw(ErrorException("Can't subtract two tangential vectors belonging to
    different tangential spaces."))
  end
end

# compare Points & vectors
=={T <: MPoint}(p::T, q::T)::Bool = all(p.value == q.value)
=={T <: MTVector}(ξ::T,ν::T)::Bool = ( sameBase(ξ,ν) && all(ξ.value==ν.value) )

function sameBase{T <: MTVector}(ξ::T, ν::T)::Bool
  if (isnull(ξ.base) || isnull(ν.base))
    return true # one base null is treated as correct
  elseif ξ.base.value == ν.base.value
    return true # if both are given and are the same
  else
    return false
  end
end
#
# proximal Maps
"""
    proxDistance(M,λ,f,p) -
  compute the proximal map with parameter λ of `distance(f,p)` for some
  fixed `MPoint` f on the `Manifold` M.
"""
function proxDistance{mT <: Manifold, T <: MPoint}(M::mT,λ::Number,f::T,p::T)::T
  exp(M,p, min(λ, distance(M,f,p))*log(M,p,f))
end

"""
    proxDistanceSquared(M,λ,f,p)
  computes the proximal map with prameter `λ` of distance^2(f,p) for some fixed
  `MPoint` f on the `Manifold` M.
"""
function proxDistanceSquared{mT <: Manifold, T <: MPoint}(M::mT,λ::Number,f::T,p::T)::T
  return exp(M,p, λ/(1+λ)*log(M,p,f) )
end
"""
    proxTuple = proxTV(M,λ,(p,q))
  Compute the proximal map prox_f(p,q) for f(p,q) = dist(p,q) with
  parameter `λ`
  # Arguments
  * `M` – a manifold
  * `(p,q)` : a tuple of size 2 containing two MPoints p and q
  * `λ` – a real value, parameter of the proximal map
  # Returns
  * `(pp,qp)` – resulting two-MPoint-Tuple of the proximal map
"""
function proxTV{mT <: Manifold, T <: MPoint}(M::mT,λ::Number, pointTuple::Tuple{T,T})::Tuple{T,T}
  step = min(0.5, λ/distance(M,pointTuple[1],pointTuple[2]))
  return (  exp(M,pointTuple[1], step*log(M,pointTuple[1],pointTuple[2])),
            exp(M,pointTuple[2], step*log(M,pointTuple[2],pointTuple[1])) )
end
"""
    proxTuple = proxTVSquared(M,λ,(p,q))
  Compute the proximal map prox_f(p,q) for f(p,q) = dist(p,q)^2 with
  parameter `λ`.
  # Arguments
  *
  * `(p,q)` : a tuple of size 2 containing two MPoints x and y
  * `λ` : a real value, parameter of the proximal map
  # OUTPUT
  * `(pr,qr)` : resulting two-MPoint-Tuple of the proximal map
"""
function proxTVSquared{mT <: Manifold, T <: MPoint}(M::mT,λ::Number, pointTuple::Tuple{T,T})::Tuple{T,T}
  step = λ/(1+2*λ)*distance(M, pointTuple[1],pointTuple[2])
  return (  exp(M, pointTuple[1], step*log(M, pointTuple[1],pointTuple[2])),
            exp(M, pointTuple[2], step*log(M, pointTuple[2],pointTuple[1])) )
end
#
# median & mean
"""
    p = mean(M,f;initialValue=[], MaxIterations=50, MinimalChange=5*10.0^(-7),
    Weights=[])

  calculates the Riemannian Center of Mass (Karcher mean) of the input data `f` with a gradient descent algorithm.
  This implementation is based on
  >B. Afsari, Riemannian Lp center of mass: Existence,
  > uniqueness, and convexity, Proc. AMS 139(2), pp.655-673, 2011.
  and
  > M. Bacak, Computing medians and means in Hadamard manifolds,
  > SIAM J. Optim., 24(3), 1542–1566, 2014.

  # Arguments
  * 'M' a manifold
  * `f` an array of `MPoint`s
  # Output
  * `p` the mean of the values from `f`
  # Optional Parameters
  * `initialValue` (`[]`) start the algorithm with a special initialisation of
    `p`, if not specified, the first value `f[1]` is used
  * `λ` (`2`) initial value for the λ of the CPP algorithm
  * `MaxIterations` (`500`) maximal number of iterations
  * `Method` (`Gradient Descent`) wether to use Gradient Descent or
    `Cyclic Proximal Point` algorithm
  * `MinimalChange` (`5*10.0^(-7)`) minimal change for the algorithm to stop
  * `Weights` (`[]`) cimpute a weigthed mean, if not specified (`[]`),
  all are choren equal, i.e. `1/n*ones(n)` for `n=length(f)`.
  ---
  ManifoldValuedImageProcessing 0.8, R. Bergmann, 2016-11-25
"""
function mean{mT <: Manifold, T <: MPoint}(M::mT, f::Vector{T}; kwargs...)::T
  # collect optional values
  # TODO: Optional values as a parameter dictionary?
  kwargs_dict = Dict(kwargs);
  x = get(kwargs_dict, "initialValue", f[1])
  Weights = get(kwargs_dict, "Weights", 1/length(f)*ones(length(f)))
  MaxIterations = get(kwargs_dict, "MaxIterations", 50)
  MinimalChange = get(kwargs_dict, "MinimalChange", 5*10.0^(-7))
  Method = get(kwargs_dict, "Method", "Gradient Descent")
  λ = get(kwargs_dict, "λ", 2)
  iter=0
  xold = x
  if Method == "Gradient Descent"
    while (  ( (distance(M,x,xold) > MinimalChange) && (iter < MaxIterations) ) || (iter == 0)  )
      xold = x
      x = exp(M,x, sum(Weights.*[log(M,x,fi) for fi in f]))
      iter += 1
    end
  elseif Method == "Cyclic Proximal Point"
    while (  ( (distance(M,x,xold) > MinimalChange) && (iter < MaxIterations) ) || (iter == 0)  )
      xold = x
      for i=1:lengthh(f)
        x = proxDistanceSquared(M,x,f[i])
      end
      iter += 1
    end
  else
    throw(ErrorException("Unknown Method to compute the mean."))
  end
  return x
end
"""
    median(M,f;initialValue=[], MaxIterations=50, MinimalChange=5*10.0^(-7),
    StepSize=1, Weights=[])

  calculates the Riemannian Center of Mass (Karcher mean) of the input data `f`
  with a gradient descent algorithm. This implementation is based on
  >B. Afsari, Riemannian Lp center of mass: Existence, uniqueness, and convexity,
  >Proc. AMS 139(2), pp.655–673, 2011.

  and
  > P. T. Fletcher, S. Venkatasubramanian, and S. Joshi: The geometric median on
  > Riemannian manifolds with application to robust atlas estimation,
  > NeuroImage 45, pp. S143–152

  # Input
  * 'M' a manifold
  * `f` an array of `MPoint`s
  # Output
  * `x` the mean of the values from `f`
  # Optional Parameters
  * `initialValue` (`[]`) start the algorithm with a special initialisation of
  `x`, if not specified, the first value `f[1]` is unsigned
  * `λ` (`2`) initial value for the λ of the CPP algorithm
  * `MaxIterations` (`500`) maximal number of iterations
  * `Method` (`Gradient Descent`) wether to use Gradient Descent or `Cyclic
      Proximal Point` algorithm
  * `MinimalChange` (`5*10.0^(-7)`) minimal change for the algorithm to stop
  * `StepSize` (`1`)
  * `Weights` (`[]`) cimpute a weigthed mean, if not specified (`[]`),
  all are choren equal, i.e. `1/n*ones(n)` for `n=length(f)`.
  ---
  ManifoldValuedImageProcessing 0.8, R. Bergmann, 2016-11-25
"""
function median{mT <: Manifold, T <: MPoint}(M::mT, f::Vector{T}; kwargs...)::T
  # collect optional values
  kwargs_dict = Dict(kwargs);
  x = get(kwargs_dict, "initialValue", f[1])
  MaxIterations = get(kwargs_dict, "MaxIterations", 50)
  MinimalChange = get(kwargs_dict, "MinimalChange", 5*10.0^(-7))
  StepSize = get(kwargs_dict, "StepSize",1)
  Weights = get(kwargs_dict, "Weights", 1/length(f)*ones(length(f)))
  Method = get(kwargs_dict, "Method", "Gradient Descent")
  iter=0
  xold = x
  if Method == "Gradient Descent"
    while (  ( (distance(M,x,xold) > MinimalChange) && (iter < MaxIterations) ) || (iter == 0)  )
      xold = x
      sumDistances = sum( Weights.*[distance(M,x,fi) for fi in f] )
      x = exp(M,x, StepSize/sumDistances * sum(Weights.* [ 1/( (distance(M,x,fi)==0)?1:distance(M,x,fi) )*log(M,x,fi) for fi in f]))
      iter += 1
    end
  elseif Method == "Cyclic Proximal Point"
    while (  ( (distance(M,x,xold) > MinimalChange) && (iter < MaxIterations) ) || (iter == 0)  )
      xold = x
      for i=1:lengthh(f)
        x = proxDistance(M,x,f[i])
      end
      iter += 1
    end
  else
    throw(ErrorException("Unknown Method to compute the mean."))
  end
  return x
end
"""
    variance(f)
  returns the variance of the set of pints on a maniofold.
"""
function variance{mT<:Manifold,T<:MPoint}(M::mT,f::Vector{T})::Number
  meanF = mean(M,f)
  return 1/( (length(f)-1)*manifoldDimension(f[1]) ) * sum( [ dist(M,meanF,fi)^2 for fi in f])
end
#
#
# Mid point and geodesics
"""
    midPoint(M,x,z)
  Compute the (geodesic) mid point of x and z.
  # Arguments
  * 'M' – a manifold
  * `p`,`q` – two `MPoint`s
  # Output
  * `m` – resulting mid point
"""
function midPoint{mT <: Manifold, T <: MPoint}(M::mT,p::T, q::T)::T
  return exp(M,p,0.5*log(p,q))
end
"""
    geodesic(M,p,q)
  return a function to evaluate the geodesic connecting `p` and `q`
  on the manifold `M`.
"""
function geodesic{mT <: Manifold, T <: MPoint}(M::mT, p::T,q::T)::Function
  return (t -> exp(M,p,t*log(M,p,q)))
end
"""
    geodesic(M,p,q,n)
  returns vector containing the equispaced n sample-values along the geodesic
  from `p`to `q` on the manifold `M`.
"""
function geodesic{mT <: Manifold, T <: MPoint}(M::mT, p::T,q::T,n::Integer)::Vector{T}
  geo = geodesic(M,p,q);
  return [geo(t) for t in linspace(0,1,n)]
end
"""
    geodesic(M,p,q,t)
  returns the point along the geodesic from `p`to `q` given by the `t`(in [0,1]) on the manifold `M`
"""
geodesic{mT <: Manifold, T <: MPoint}(M::mT,p::T,q::T,t::Number)::T = geodesic(p,q)(t)
"""
    geodesic(Mp,q,T)
  returns vector containing the MPoints along the geodesic from `p`to `q` on
  the manfiold `M` specified within the vector `T` (of numbers between 0 and 1).
"""
function geodesic{mT <: Manifold, T <: MPoint, S <: Number}(M::mT, p::T,q::T,v::Vector{S})::Vector{T}
  geo = geodesic(M,p,q);
  return [geo(t) for t in v]
end
#
# fallback functions for not yet implemented cases – for example also for the
# cases where you take the wrong manifolg for certain points
"""
    addNoise(M,P,σ)
  adds noise of standard deviation `σ` to the MPoint `p` on the manifold `M`.
"""
function addNoise{mT <: Manifold, T <: MPoint}(M::mT,P::T,σ::Number)::T
  sig1 = string( typeof(P) )
  sig2 = string( typeof(σ) )
  sig3 = string( typeof(M) )
  throw( ErrorException(" addNoise – not Implemented for Point $sig1 and $sig2 on the manifold $sig3.") )
end
"""
    distance(M,p,q)
  computes the gedoesic distance between two points `p`and `q`
  on a manifold `M`.
"""
function distance{mT <: Manifold, T <: MPoint}(M::mT, p::T,q::T)::Number
  sig1 = string( typeof(p) )
  sig2 = string( typeof(q) )
  sig3 = string( typeof(M) )
  throw( ErrorException(" Distance – not Implemented for the two points $sig1 and $sig2 on the manifold $sig3." ) )
end
"""
    dot(M,ξ,ν)
  computes the inner product of two tangential vectors ξ=ξp and ν=νp in TpM
  of p on the manifold `M`.
"""
function dot{mT <: Manifold, T <: MTVector}(M::mT, ξ::T, ν::T)::Number
  sig1 = string( typeof(ξ) )
  sig2 = string( typeof(ν) )
  sig3 = string( typeof(M) )
  throw( ErrorException(" Dot – not Implemented for the two tangential vectors $sig1 and $sig2 on the manifold $sig3." ) )
end
"""
    exp(M,p,ξ)
  computes the exponential map at p for the tangential vector ξ=ξp
  on the manifold `M`.
"""
function exp{mT<:Manifold, T<:MPoint, S<:MTVector}(M::mT, p::T, ξ::S)::MPoint
  sig1 = string( typeof(p) )
  sig2 = string( typeof(ξ) )
  sig3 = string( typeof(M) )
  throw( ErrorException(" Exp – not Implemented for Point $sig1 and tangential vector $sig2 on the manifold $sig3." ) )
end
"""
    log(M,p,q)
  computes the tangential vector at p whose unit speed geodesic reaches q after time T = distance(Mp,q) (not t) (note that the geodesic above is [0,1]
  parametrized).
"""
function log{mT<:Manifold, T<:MPoint}(M::mT,p::T,q::T)::MTVector
  sig1 = string( typeof(p) )
  sig2 = string( typeof(q) )
  sig3 = string( typeof(M) )
  throw( ErrorException(" Log – not Implemented for Points $sig1 and $sig2 on the manifold $sig3.") )
end
"""
    manifoldDimension(M) or manifoldDimension(p)
  returns the dimension of the manifold `M` the point p belongs to.
"""
function manifoldDimension{T<:MPoint}(p::T)::Integer
  sig1 = string( typeof(p) )
  throw( ErrorException(" Not Implemented for manifodl points $sig1 " ) )
end
function manifoldDimension{T<:Manifold}(M::T)::Integer
  sig1 = string( typeof(M) )
  throw( ErrorException(" Not Implemented for manifold $sig1 " ) )
end
"""
    norm(M,ξ)
  computes the lenth of a tangential vector
"""
function norm{mT<:Manifold, T<:MTVector}(M::mT,ξ::T)::Number
  sig1 = string( typeof(ξ) )
  throw( ErrorException(" Not Implemented for types $sig1 " ) )
end
