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
    y = mean(M,x;initialValue=[], MaxIterations=50, MinimalChange=5*10.0^(-7),
    Weights=[])
calculates the Riemannian Center of Mass (Karcher mean) of the input data `f` with a gradient descent algorithm.
This implementation is based on
>B. Afsari, Riemannian Lp center of mass: Existence,
> uniqueness, and convexity, Proc. AMS 139(2), pp.655-673, 2011.
and
> M. Bacak, Computing medians and means in Hadamard manifolds,
> SIAM J. Optim., 24(3), 1542–1566, 2014.
# Arguments
* `M` a manifold
* `x` an array of `MPoint`s
# Output
* `y` the mean of the values from `x`
# Optional Parameters
the default values are given in brackets
* `initialValue` : (`[]`) start the algorithm with a special initialisation of
`y`, if not specified, the first value `x[1]` is used
* `λ` : (`2`) initial value for the λ of the CPP algorithm
* `maxIterations` (`500`) maximal number of iterations
* `method` (:Gradient Descent) wether to use Gradient Descent or
  the Cyclic Proximal Point (:CyclicProximalPoint) algorithm
* `minimalChange` (`5*10.0^(-7)`) minimal change for the algorithm to stop
* `weights` (`[]`) cimpute a weigthed mean, if not specified (`[]`),
  all are choren equal, i.e. `1/n*ones(n)` for `n=length(x)`.
  ---
  ManifoldValuedImageProcessing 0.8, R. Bergmann, 2016-11-25
"""
function mean(M::mT, x::Vector{T}; kwargs...)::T where {mT <: Manifold, T <: MPoint}
  # collect optional values
  # TODO: Optional values as a parameter dictionary?
  kwargs_dict = Dict(kwargs);
  y = get(kwargs_dict, "initialValue", x[1])
  Weights = get(kwargs_dict, "weights", 1/length(x)*ones(length(x)))
  MaxIterations = get(kwargs_dict, "maxIterations", 50)
  MinimalChange = get(kwargs_dict, "minimalChange", 5*10.0^(-7))
  Method = get(kwargs_dict, "method", useGradientDescent())
  λ = get(kwargs_dict, "λ", 2)
  return mean_(M,x,y,Weights,λ,MinimalChange,MaxIterations,Method)
end
function mean_(M,x,y,w,λ,mC,mI,::useGradientDescent)
  iter = 0; yold = y
  while (  ( (distance(M,y,yold) > mC) && (iter < mI) ) || (iter == 0)  )
    yold = y
    y = exp(M,y, sum(w.*[log(M,y,xi) for xi in x]))
    iter += 1
  end
  return y
end
function mean_(M,x,y,w,λ,mC,mI,::useCyclicProximalPoint)
  while (  ( (distance(M,y,yold) > mC) && (iter < mI) ) || (iter == 0)  )
    yold = y
    iter += 1
    λi = λ/iter;
    for i=1:lengthh(x)
      y = proxDistanceSquared(M,λi*w[i],y,x[i])
    end
  end
  return y
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
  the default values are given in brackets
  * `initialValue` : (`[]`) start the algorithm with a special initialisation of
  `y`, if not specified, the first value `x[1]` is used
  * `λ` : (`2`) initial value for the λ of the CPP algorithm
  * `maxIterations` (`500`) maximal number of iterations
  * `method` (SubGradientDescent) wether to use Gradient Descent or
    the Cyclic Proximal Point (:CyclicProximalPoint) algorithm
  * `minimalChange` (`5*10.0^(-7)`) minimal change for the algorithm to stop
  * `stepSize` a step size for the subgradient descent
  * `weights` (`[]`) cimpute a weigthed mean, if not specified (`[]`),
    all are chosen equal, i.e. `1/n*ones(n)` for `n=length(x)`.
"""
function median(M::mT, f::Vector{T}; kwargs...)::T where {mT <: Manifold, T <: MPoint}
  # collect optional values
  kwargs_dict = Dict(kwargs);
  y = get(kwargs_dict, "initialValue", x[1])
  Weights = get(kwargs_dict, "weights", 1/length(x)*ones(length(x)))
  StepSize = get(kwargs_dict, "stepSize",1)
  λ = get(kwargs_dict, "λ", 2)
  MinimalChange = get(kwargs_dict, "minimalChange", 5*10.0^(-7))
  MaxIterations = get(kwargs_dict, "maxIterations", 50)
  Method = get(kwargs_dict, "method", useSubgradientDescent())
  return median_(M,x,y,Weights,StepSize,λ,MinimalChange,MaxIterations,Method)
end
function median_(M,x,y,w,s,λ,mC,mI,::useSubgradientDescent)
  iter=0;
  yold = y;
  while (  ( (distance(M,y,yold) > mC) && (iter < mI) ) || (iter == 0)  )
    yold = y
    sumDistances = sum( w.*[distance(M,y,ξ) for ξ in x] )
    y = exp(M,y, s/sumDistances * sum(w.* [ 1/( (distance(M,y,ξ)==0) ? 1 : distance(M,y,ξ) )*log(M,y,ξ) for ξ in x]))
    iter += 1
  end
  return y
end
function median_(M,x,y,w,s,λ,mC,mI,::useCyclicProximalPoint)
  iter=0
  yold = y
  while (  ( (distance(M,y,yold) > mC) && (iter < mI) ) || (iter == 0)  )
    iter += 1
    λi = λ/iter;
    yold = y
    for i=1:lengthh(x)
      x = proxDistance(M,λi*w[i],y,x[i])
    end
  end
  return y
end

"""
    variance(x)
  returns the variance of the vector `x` of points on a maniofold.
"""
function variance(M::mT,x::Vector{T}) where {mT<:Manifold,T<:MPoint}
  meanX = mean(M,x)
  return 1/( (length(x)-1)*manifoldDimension(M) ) * sum( [ dist(M,meanX,xi)^2 for xi in x])
end
