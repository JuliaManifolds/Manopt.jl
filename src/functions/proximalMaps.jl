#
# Manopt.jl – Proximal maps
#
# This file provides several proximal maps on manifolds or on small
# product manifolds, like M^2
#
# ---
# Manopt.jl - R. Bergmann – 2017-07-06

export proxDistance, proxTV, proxParallelTV, proxTV2, proxCollaborativeTV

@doc doc"""
    y = proxDistance(M,λ,f,x [,p=2])

compute the proximal map $\operatorname{prox}_{\lambda\varphi}$ with
parameter λ of $\varphi(x) = \frac{1}{p}d_{\mathcal M}^p(f,x)$.

# Input
* `M` – a [`Manifold`](@ref) $\mathcal M$
* `λ` – the prox parameter
* `f` – an [`MPoint`](@ref) $f\in\mathcal M$ (the data)
* `x` – the argument of the proximal map

# Optional argument
* `p` – (`2`) exponent of the distance.

# Ouput
* `y` – the result of the proximal map of $\varphi$
"""
function proxDistance(M::mT,λ::Number,f::T,x::T,p::Int=2) where {mT <: Manifold, T <: MPoint}
  d = distance(M,f,x)
  if p==2
    t =  λ/(1+λ);
  elseif p==1
    if λ < d
      t = λ/d;
    else
      t = 1.;
    end
  else
      throw(ErrorException(
        "Proximal Map of distance(M,f,x) not implemented for p=$(p) (requires p=1 or 2)"
      ))
  end
  return exp(M,x,log(M,x,f),t);
end
@doc doc"""
    (y1,y2) = proxTV(M,λ,(x1,x2) [,p=1])

Compute the proximal map $\operatorname{prox}_{\lambda\varphi}$ of
$\varphi(x,y) = d_{\mathcal M}^p(x,y)$ with
parameter `λ`.

# Input
* `M` – a [`Manifold`](@ref)
* `λ` – a real value, parameter of the proximal map
* `(x1,x2)` – a tuple of two [`MPoint`](@ref)s,

# Optional
(default is given in brackets)
* `p` – (1) exponent of the distance of the TV term

# Ouput
* `(y1,y2)` – resulting tuple of [`MPoint`](@ref) of the
  $\operatorname{prox}_{\lambda\varphi}($ `(x1,x2)` $)$
"""
function proxTV(M::mT,λ::Number, pointTuple::Tuple{P,P},p::Int=1)::Tuple{P,P} where {mT <: Manifold, P <: MPoint}
  x1 = pointTuple[1];
  x2 = pointTuple[2];
  d = distance(M,x1,x2);
  if p==1
    t = min(0.5, λ/d);
  elseif p==2
    t = λ/(1+2*λ);
  else
    throw(ErrorException(
      "Proximal Map of TV(M,x1,x2,p) not implemented for p=$(p) (requires p=1 or 2)"
    ))
  end
  return (  exp(M, x1, log(M, x1, x2), t), exp(M, x2, log(M, x2, x1), t)  );
end
@doc doc"""
    ξ = proxTV(M,λ,x [,p=1])

compute the proximal maps $\operatorname{prox}_{\lambda\varphi}$ of
all forward differences orrucirng in the power manifold array, i.e.
$\varphi(xi,xj) = d_{\mathcal M}^p(xi,xj)$ with `xi` and `xj` are array
elemets of `x` and `j = i+e_k`, where `e_k` is the $k$th unitvector.
The parameter `λ` is the prox parameter.

# Input
* `M` – a [`Manifold`](@ref)
* `λ` – a real value, parameter of the proximal map
* `x` – a [`PowPoint`](@ref).

# Optional
(default is given in brackets)
* `p` – (1) exponent of the distance of the TV term

# Ouput
* `y` – resulting of [`PowPoint`](@ref) with all mentioned proximal
  points evaluated (in a cylic order).
"""
function proxTV(M::Power, λ::Number, x::PowPoint,p::Int=1)::PowPoint
  R = CartesianIndices(M.powerSize)
  d = length(M.powerSize)
  maxInd = Tuple(last(R))
  y = copy(x)
  for k in 1:d # for all directions
    ek = CartesianIndex(ntuple(i  ->  (i==k) ? 1 : 0, d) ) #k th unit vector
    for l in 0:1
      for i in R # iterate over all pixel
        if (i[k] % 2) == l
          I = [i.I...] # array of index
          J = I .+ 1 .* (1:d .== k) #i + e_k is j
          if all( J .<= maxInd ) # is this neighbor in range?
            j = CartesianIndex(J...) # neigbbor index as Cartesian Index
            (y[i],y[j]) = proxTV( M.manifold,λ,(y[i],y[j]),p) # Compute TV on these
          end
        end
      end # i in R
    end # even odd
  end # directions
  return y
end
@doc doc"""
    ξ = proxParallelTV(M,λ,x [,p=1])

compute the proximal maps $\operatorname{prox}_{\lambda\varphi}$ of
all forward differences orrucirng in the power manifold array, i.e.
$\varphi(xi,xj) = d_{\mathcal M}^p(xi,xj)$ with `xi` and `xj` are array
elemets of `x` and `j = i+e_k`, where `e_k` is the $k$th unitvector.
The parameter `λ` is the prox parameter.

# Input
* `M`     – a [`Power`](@ref) manifold
* `λ`     – a real value, parameter of the proximal map
* `x`     – a [`PowPoint`](@ref).

# Optional
(default is given in brackets)
* `p` – (`1`) exponent of the distance of the TV term

# Ouput
* `y`  – resulting of Array [`PowPoint`](@ref)s with all mentioned proximal
  points evaluated (in a parallel within the arrays elements).

*See also* [`proxTV`](@ref)
"""
function proxParallelTV(M::Power, λ::Number, x::Array{PowPoint{P,N},1}, p::Int=1)::Array{PowPoint{P,N},1} where {P <: MPoint, N}
  R = CartesianIndices(getValue(x[1]))
  d = ndims(getValue(x[1]))
  if length(x) != 2*d
    throw(ErrorException("The number of inputs from the array ($(length(x))) has to be twice the data dimensions ($(d))."))
  end
  maxInd = Tuple(last(R))
  # create an array for even/odd splitted proxes along every dimension
  y = reshape(deepcopy(x),d,2)
  x = reshape(x,d,2)
  for k in 1:d # for all directions
    ek = CartesianIndex(ntuple(i  ->  (i==k) ? 1 : 0, d) ) #k th unit vector
    for l in 0:1 # even odd
      for i in R # iterate over all pixel
        if (i[k] % 2) == l
          I = [i.I...] # array of index
          J = I .+ 1 .* (1:d .== k) #i + e_k is j
          if all( J .<= maxInd ) # is this neighbor in range?
            j = CartesianIndex(J...) # neigbbor index as Cartesian Index
            # parallel means we apply each (direction even/odd) to a seperate copy of the data.
            (y[k,l+1][i],y[k,l+1][j]) = proxTV( M.manifold,λ,(x[k,l+1][i],x[k,l+1][j]),p) # Compute TV on these
          end
        end
      end # i in R
    end # even odd
  end # directions
  return y[:] # return as onedimensional array
end
@doc doc"""
    (y1,y2,y3) = proxTV2(M,λ,(x1,x2,x3),[p=1], kwargs...)

Compute the proximal map $\operatorname{prox}_{\lambda\varphi}$ of
$\varphi(x_1,x_2,x_3) = d_{\mathcal M}^p(c(x_1,x_3),x_2)$ with
parameter `λ`>0, where $c(x,z)$ denotes the mid point of a shortest
geodesic from `x1` to `x3` that is closest to `x2`.

# Input

* `M`          – a manifold
* `λ`          – a real value, parameter of the proximal map
* `(x1,x2,x3)` – a tuple of three [`MPoint`](@ref)s

* `p` – (`1`) exponent of the distance of the TV term

# Optional
`kwargs...` – parameters for the internal [`subGradientMethod`](@ref)
    (if `M` is neither `Euclidean` nor `Circle`, since for these a closed form
    is given)

# Output
* `(y1,y2,y3)` – resulting tuple of [`MPoint`](@ref)s of the proximal map
"""
function proxTV2(M::mT,λ,pointTuple::Tuple{P,P,P},p::Int=1;
  stoppingCriterion::StoppingCriterion = stopAfterIteration(5),
  kwargs...)::Tuple{P,P,P} where {mT <: Manifold, P <: MPoint}
  if p != 1
    throw(ErrorException(
      "Proximal Map of TV2(M,λ,pT,p) not implemented for p=$(p) (requires p=1) on general manifolds."
    ))
  end
  PowX = PowPoint([pointTuple...])
  PowM = Power(M,(3,))
  xInit = PowX
  F(x) = 1/2*distance(PowM,PowX,x)^2 + λ*costTV2(PowM,x)
  ∂F(x) = log(PowM,x,PowX) + λ*gradTV2(PowM,x)
  xR = subGradientMethod(PowM,F,∂F,xInit;stoppingCriterion=stoppingCriterion, kwargs...)
  return (getValue(xR)...,)
end
function proxTV2(M::Circle,λ,pointTuple::Tuple{S1Point,S1Point,S1Point},p::Int=1)::Tuple{S1Point,S1Point,S1Point}
  w = [1., -2. ,1. ]
  x = [getValue.(pointTuple)...]
  if p==1 # Theorem 3.5 in Bergmann, Laus, Steidl, Weinmann, 2014.
    m = min(   λ, abs(  symRem( sum( x .* w  ) )  )/(dot(w,w))   )
    s = sign( symRem(sum(x .* w)) )
    return Tuple( S1Point.( symRem.( x  .-  m .* s .* w ) ) )
  elseif p==2 # Theorem 3.6 ibd.
    t = λ * symRem( sum( x .* w ) ) / (1 + λ*dot(w,w) )
    return Tuple(  S1Point.( symRem.( x - t.*w ) )  )
  else
    throw(ErrorException(
      "Proximal Map of TV2(Circle,λ,pT,p) not implemented for p=$(p) (requires p=1 or 2)"
    ))
  end
end
function proxTV2(M::Euclidean,λ,pointTuple::Tuple{RnPoint,RnPoint,RnPoint},p::Int=1)::Tuple{RnPoint,RnPoint,RnPoint}
  w = [1., -2. ,1. ]
  x = [getValue.(pointTuple)...]
  if p==1 # Example 3.2 in Bergmann, Laus, Steidl, Weinmann, 2014.
    m = min.(Ref(λ),  abs.( x .* w  ) / (dot(w,w))   )
    s = sign.( sum(x .* w) )
    return Tuple( RnPoint.( x  .-  m .* s .* w ) )
  elseif p==2 # Theorem 3.6 ibd.
    t = λ * sum( x .* w ) / (1 + λ*dot(w,w) )
    return Tuple(  RnPoint.( x - t.*w ) )
  else
    throw(ErrorException(
      "Proximal Map of TV2(Euclidean,λ,pT,p) not implemented for p=$(p) (requires p=1 or 2)"
    ))
  end
end
@doc doc"""
    ξ = proxTV2(M,λ,x,[p])

compute the proximal maps $\operatorname{prox}_{\lambda\varphi}$ of
all centered second order differences orrucirng in the power manifold array, i.e.
$\varphi(x_k,x_i,x_j) = d_2(x_k,x_i.x_j)$, where $k,j$ are backward and forward
neighbors (along any dimension in the array of `x`).
The parameter `λ` is the prox parameter.

# Input
* `M` – a [`Manifold`](@ref)
* `λ` – a real value, parameter of the proximal map
* `x` – a [`PowPoint`](@ref).

# Optional
(default is given in brackets)
* `p` – (`1`) exponent of the distance of the TV term

# Ouput
* `y` – resulting of [`PowPoint`](@ref) with all mentioned proximal points
  evaluated (in a cylic order).
"""
function proxTV2(M::Power, λ::Number, x::PowPoint,p::Int=1)::PowPoint
  R = CartesianIndices(M.powerSize)
  d = length(size(x))
  minInd = [first(R).I...]
  maxInd = [last(R).I...]
  y = copy(x)
  for k in 1:d # for all directions
    for l in 0:1
      for i in R # iterate over all pixel
        if (i[k] % 3) == l
          I = [i.I...] # array of index
          JForward = I .+ 1 .* (1:d .== k) #i + e_k
          JBackward = I .+ 1 .* (1:d .== k) # i - e_k
          if all( JForward .<= maxInd ) && all( JBackward .>= minInd)
            jForward = CartesianIndex{d}(JForward...) # neigbbor index as Cartesian Index
            jBackward = CartesianIndex{d}(JForward...) # neigbbor index as Cartesian Index
            (y[jBackward], y[i], y[jForward]) = 
              proxTV2( M.manifold, λ, (y[jBackward], y[i], y[jForward]),p) # Compute TV on these
          end
        end # if mod 3
      end # i in R
    end # for mod 3
  end # directions
  return y
end
@doc doc"""
    proxCollaborativeTV(M,λ,x [,p=2,q=1])

compute the prox of the collaborative TV prox for x on the [`Power`](@ref)
manifold, i.e. of the function

```math
F^q(x) = \sum_{i\in\mathcal G}
  \Bigl( \sum_{j\in\mathcal I_i}
    \sum_{k=1^d} \lVert X_{i,j}\rVert_x^p\Bigr)^\frac{q/p},
```

where $\mathcal G$ is the set of indices for $x\in\mathcal M$ and $\mathcal I_i$
is the set of its forward neighbors.
This is adopted from the paper by Duran, Möller, Sbert, Cremers:
_Collaborative Total Variation: A General Framework for Vectorial TV Models_
(arxiv: [1508.01308](https://arxiv.org/abs/1508.01308)), where the most inner
norm is not on a manifold but on a vector space, see their Example 3 for
details.
"""
function proxCollaborativeTV(N::Power,λ::Float64,x::PowPoint,Ξ::PowTVector,p::Float64=2.,q::Float64=1.)
  # Ξ = forwardLogs(M,x)
  if length(size(x)) == 1
    d = 1
    s = 1
    iRep = 1
  else
    d = size(x)[end]
    s = length(size(x))-1
    if s != d
      throw( ErrorException( "the last dimension ($(d)) has to be equal to the number of the previous ones ($(s)) but its not." ))
    end
    iRep = [Integer.(ones(d))...,d]
  end
  if q==1 # Example 3 case 2
    if p==1
      normΞ = norm.(Ref(N.manifold), getValue(x), getValue(Ξ) )
      return PowTVector( max.(normΞ .- λ, 0.) ./ ( (normΞ .== 0) .+ normΞ )  .*  getValue(Ξ) )
    elseif p==2 # Example 3 case 3
      norms = sqrt.( sum( norm.(Ref(N.manifold),getValue(x),getValue(Ξ)).^2, dims=d+1) )
      normΞ = repeat(norms,inner=iRep)
      # if the norm is zero add 1 to avoid division by zero, also then the
      # nominator is already (max(-λ,0) = 0) so it stays zero then
      return PowTVector( max.(normΞ .- λ, 0.) ./ ( (normΞ .== 0) .+ normΞ )  .*  getValue(Ξ) )
    else
      throw( ErrorException("The case p=$p, q=$q is not yet implemented"))
    end
  elseif q==Inf
    if p==2
      norms = sqrt.( sum( norm.(Ref(N.manifold),getValue(x),getValue(Ξ)).^2, dims=d+1) )
      normΞ = repeat(norms,inner=iRep)
    elseif p==1
      norms = sum( norm.(Ref(N.manifold),getValue(x),getValue(Ξ)), dims=d+1)
      normΞ = repeat(norms,inner=iRep)
    elseif p==Inf
      normΞ = norm.(Ref(N.manifold),getValue(x),getValue(Ξ))
    else
      throw( ErrorException("The case p=$p, q=$q is not yet implemented"))
    end
    return PowTVector(
      λ .* getValue(Ξ) ./ max.(Ref(λ), normΞ)
    )
  end # end q
  throw( ErrorException("The case p=$p, q=$q is not yet implemented"))
end
proxCollaborativeTV(N::Power,λ::Float64,x::PowPoint,Ξ::PowTVector,p::Int,q::Float64=1.) = proxCollaborativeTV(N,λ,x,Ξ,Float64(p),q)
proxCollaborativeTV(N::Power,λ::Float64,x::PowPoint,Ξ::PowTVector,p::Float64,q::Int) = proxCollaborativeTV(N,λ,x,Ξ,p,Float64(q))
proxCollaborativeTV(N::Power,λ::Float64,x::PowPoint,Ξ::PowTVector,p::Int,q::Int) = proxCollaborativeTV(N,λ,x,Ξ,Float64(p),Float64(q))
