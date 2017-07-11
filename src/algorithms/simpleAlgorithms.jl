#
# Manopt.jl – simple algorithms
#
# A collection of simple algorithms that might be helpful
#
# ---
# Manopt.jl – Ronny Bergmann – 2017-07-06
import Base: mean, median
export mean, median, variance

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
