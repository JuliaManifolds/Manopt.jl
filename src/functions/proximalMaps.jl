@doc raw"""
    y = proxDistance(M,λ,f,x [,p=2])

compute the proximal map $\operatorname{prox}_{\lambda\varphi}$ with
parameter λ of $\varphi(x) = \frac{1}{p}d_{\mathcal M}^p(f,x)$.

# Input
* `M` – a [`Manifold`](@ref) $\mathcal M$
* `λ` – the prox parameter
* `f` – a point $f ∈ \mathcal M$ (the data)
* `x` – the argument of the proximal map

# Optional argument
* `p` – (`2`) exponent of the distance.

# Ouput
* `y` – the result of the proximal map of $\varphi$
"""
function proxDistance(M::MT,λ::Number,f::T,x::T,p::Int=2) where {MT <: Manifold, T}
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
@doc raw"""
    (y1,y2) = proxTV(M,λ,(x1,x2) [,p=1])

Compute the proximal map $\operatorname{prox}_{\lambda\varphi}$ of
$\varphi(x,y) = d_{\mathcal M}^p(x,y)$ with
parameter `λ`.

# Input
* `M` – a [`Manifold`](@ref)
* `λ` – a real value, parameter of the proximal map
* `(x1,x2)` – a tuple of two points,

# Optional
(default is given in brackets)
* `p` – (1) exponent of the distance of the TV term

# Ouput
* `(y1,y2)` – resulting tuple of points of the
  $\operatorname{prox}_{\lambda\varphi}($ `(x1,x2)` $)$
"""
function proxTV(M::mT,λ::Number, pointTuple::Tuple{T,T}, p::Int=1) where {mT <: Manifold,T}
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
@doc raw"""
    ξ = proxTV(M,λ,x [,p=1])

compute the proximal maps $\operatorname{prox}_{\lambda\varphi}$ of
all forward differences orrucirng in the power manifold array, i.e.
$\varphi(xi,xj) = d_{\mathcal M}^p(xi,xj)$ with `xi` and `xj` are array
elemets of `x` and `j = i+e_k`, where `e_k` is the $k$th unitvector.
The parameter `λ` is the prox parameter.

# Input
* `M` – a [`Manifold`](@ref)
* `λ` – a real value, parameter of the proximal map
* `x` – a point.

# Optional
(default is given in brackets)
* `p` – (1) exponent of the distance of the TV term

# Ouput
* `y` – resulting  point containinf with all mentioned proximal
  points evaluated (in a cylic order).
"""
function proxTV(M::PowerManifold{N,T}, λ, x, p::Int=1) where {N <: Manifold, T}
    power_size = [T.parameters...]
    R = CartesianIndices(Tuple(power_size))
    d = length(power_size)
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
@doc raw"""
    ξ = proxParallelTV(M,λ,x [,p=1])

compute the proximal maps $\operatorname{prox}_{\lambda\varphi}$ of
all forward differences orrucirng in the power manifold array, i.e.
$\varphi(xi,xj) = d_{\mathcal M}^p(xi,xj)$ with `xi` and `xj` are array
elemets of `x` and `j = i+e_k`, where `e_k` is the $k$th unitvector.
The parameter `λ` is the prox parameter.

# Input
* `M`     – a `PowerManifold` manifold
* `λ`     – a real value, parameter of the proximal map
* `x`     – a point

# Optional
(default is given in brackets)
* `p` – (`1`) exponent of the distance of the TV term

# Ouput
* `y`  – resulting Array of points with all mentioned proximal
  points evaluated (in a parallel within the arrays elements).

*See also* [`proxTV`](@ref)
"""
function proxParallelTV(M::PowerManifold, λ, x::Array{T,1}, p::Int=1) where {T}
  R = CartesianIndices(x[1])
  d = ndims(x[1])
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
@doc raw"""
    (y1,y2,y3) = proxTV2(M,λ,(x1,x2,x3),[p=1], kwargs...)

Compute the proximal map $\operatorname{prox}_{\lambda\varphi}$ of
$\varphi(x_1,x_2,x_3) = d_{\mathcal M}^p(c(x_1,x_3),x_2)$ with
parameter `λ`>0, where $c(x,z)$ denotes the mid point of a shortest
geodesic from `x1` to `x3` that is closest to `x2`.

# Input

* `M`          – a manifold
* `λ`          – a real value, parameter of the proximal map
* `(x1,x2,x3)` – a tuple of three points

* `p` – (`1`) exponent of the distance of the TV term

# Optional
`kwargs...` – parameters for the internal [`subGradientMethod`](@ref)
    (if `M` is neither `Euclidean` nor `Circle`, since for these a closed form
    is given)

# Output
* `(y1,y2,y3)` – resulting tuple of points of the proximal map
"""
function proxTV2(M::mT,λ,pointTuple::Tuple{T,T,T},p::Int=1;
  stoppingCriterion::StoppingCriterion = stopAfterIteration(5),
  kwargs...) where {mT <: Manifold,T}
  if p != 1
    throw(ErrorException(
      "Proximal Map of TV2(M,λ,pT,p) not implemented for p=$(p) (requires p=1) on general manifolds."
    ))
  end
  PowX = [pointTuple...]
  PowM = PowerManifold(M, NestedPowerRepresentation(), 3)
  xInit = PowX
  F(x) = 1/2*distance(PowM,PowX,x)^2 + λ*costTV2(PowM,x)
  ∂F(x) = log(PowM,x,PowX) + λ*gradTV2(PowM,x)
  xR = subGradientMethod(PowM,F,∂F,xInit;stoppingCriterion=stoppingCriterion, kwargs...)
  return (xR...,)
end
function proxTV2(M::Circle,λ,pointTuple::Tuple{T,T,T},p::Int=1) where {T}
  w = [1., -2. ,1. ]
  x = [pointTuple...]
  if p==1 # Theorem 3.5 in Bergmann, Laus, Steidl, Weinmann, 2014.
    m = min( λ, abs(  sym_rem( sum( x .* w  ) ) )/(dot(w,w))   )
    s = sign( sym_rem(sum(x .* w)) )
    return Tuple(  sym_rem.( x  .-  m .* s .* w ) )
  elseif p==2 # Theorem 3.6 ibd.
    t = λ * sym_rem( sum( x .* w ) ) / (1 + λ*dot(w,w) )
    return Tuple( sym_rem.( x - t.*w )  )
  else
    throw(ErrorException(
      "Proximal Map of TV2(Circle,λ,pT,p) not implemented for p=$(p) (requires p=1 or 2)"
    ))
  end
end
function proxTV2(M::Euclidean,λ,pointTuple::Tuple{T,T,T},p::Int=1) where {T}
  w = [1., -2. ,1. ]
  x = [pointTuple...]
  if p==1 # Example 3.2 in Bergmann, Laus, Steidl, Weinmann, 2014.
    m = min.(Ref(λ),  abs.( x .* w  ) / (dot(w,w))   )
    s = sign.( sum(x .* w) )
    return x  .-  m .* s .* w
  elseif p==2 # Theorem 3.6 ibd.
    t = λ * sum( x .* w ) / (1 + λ*dot(w,w) )
    return x - t.*w
  else
    throw(ErrorException(
      "Proximal Map of TV2(Euclidean,λ,pT,p) not implemented for p=$(p) (requires p=1 or 2)"
    ))
  end
end
@doc raw"""
    ξ = proxTV2(M,λ,x,[p])

compute the proximal maps $\operatorname{prox}_{\lambda\varphi}$ of
all centered second order differences orrucirng in the power manifold array, i.e.
$\varphi(x_k,x_i,x_j) = d_2(x_k,x_i.x_j)$, where $k,j$ are backward and forward
neighbors (along any dimension in the array of `x`).
The parameter `λ` is the prox parameter.

# Input
* `M` – a [`Manifold`](@ref)
* `λ` – a real value, parameter of the proximal map
* `x` – a points.

# Optional
(default is given in brackets)
* `p` – (`1`) exponent of the distance of the TV term

# Ouput
* `y` – resulting point with all mentioned proximal points
  evaluated (in a cylic order).
"""
function proxTV2(M::PowerManifold{N,T}, λ, x, p::Int=1) where {N,T}
  power_size = [T.parameters...]
  R = CartesianIndices(power_size)
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
@doc raw"""
    proxCollaborativeTV(M,λ,x [,p=2,q=1])

compute the prox of the collaborative TV prox for x on the `PowerManifold`
manifold, i.e. of the function

```math
F^q(x) = \sum_{i ∈ \mathcal G}
  \Bigl( \sum_{j ∈ \mathcal I_i}
    \sum_{k=1^d} \lVert X_{i,j}\rVert_x^p\Bigr)^\frac{q/p},
```

where $\mathcal G$ is the set of indices for $x ∈ \mathcal M$ and $\mathcal I_i$
is the set of its forward neighbors.
This is adopted from the paper by Duran, Möller, Sbert, Cremers:
_Collaborative Total Variation: A General Framework for Vectorial TV Models_
(arxiv: [1508.01308](https://arxiv.org/abs/1508.01308)), where the most inner
norm is not on a manifold but on a vector space, see their Example 3 for
details.
"""
function proxCollaborativeTV(N::PowerManifold, λ, x, Ξ,p=2.,q=1.)
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
      normΞ = norm.(Ref(N.manifold), x, Ξ)
      return  max.(normΞ .- λ, 0.) ./ ( (normΞ .== 0) .+ normΞ )  .*  Ξ
    elseif p==2 # Example 3 case 3
      norms = sqrt.( sum( norm.(Ref(N.manifold),x,Ξ).^2, dims=d+1))
      normΞ = repeat(norms,inner=iRep)
      # if the norm is zero add 1 to avoid division by zero, also then the
      # nominator is already (max(-λ,0) = 0) so it stays zero then
      return  max.(normΞ .- λ, 0.) ./ ( (normΞ .== 0) .+ normΞ )  .*  Ξ
    else
      throw( ErrorException("The case p=$p, q=$q is not yet implemented"))
    end
  elseif q==Inf
    if p==2
      norms = sqrt.( sum( norm.(Ref(N.manifold),x,Ξ).^2, dims=d+1))
      normΞ = repeat(norms,inner=iRep)
    elseif p==1
      norms = sum( norm.(Ref(N.manifold), x, Ξ), dims=d+1)
      normΞ = repeat(norms,inner=iRep)
    elseif p==Inf
      normΞ = norm.(Ref(N.manifold),x,Ξ)
    else
      throw( ErrorException("The case p=$p, q=$q is not yet implemented"))
    end
    return (λ .* Ξ) ./ max.(Ref(λ), normΞ)
  end # end q
  throw( ErrorException("The case p=$p, q=$q is not yet implemented"))
end
proxCollaborativeTV(N::PowerManifold, λ, x, Ξ, p::Int, q::Float64=1.) = proxCollaborativeTV(N,λ,x,Ξ,Float64(p),q)
proxCollaborativeTV(N::PowerManifold, λ, x, Ξ, p::Float64, q::Int) = proxCollaborativeTV(N,λ,x,Ξ,p,Float64(q))
proxCollaborativeTV(N::PowerManifold, λ, x, Ξ, p::Int, q::Int) = proxCollaborativeTV(N,λ,x,Ξ,Float64(p),Float64(q))
