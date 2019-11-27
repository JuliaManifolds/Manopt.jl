export DxGeo, DyGeo, DxExp,DξExp, DxLog, DyLog
export DforwardLogs
@doc doc"""
    DxGeo(M,x,y,t,η)
computes $D_xg(t;x,y)[\eta]$.

# See also
 [`DyGeo`](@ref), [`jacobiField`](@ref)
"""
DxGeo(M::mT,x::P,y::P,t::Number,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector} = jacobiField(M,x,y,t,η,βDgx)
@doc doc"""
    DyGeo(M,x,y,t,η)
computes $D_yg(t;x,y)[\eta]$.

# See also
 [`DxGeo`](@ref), [`jacobiField`](@ref)
"""
DyGeo(M::mT,x::P,y::P,t::Number,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector} = jacobiField(M,y,x,1-t,η,βDgx)
@doc doc"""
    DxExp(M,x,ξ,η)
computes $D_x\exp_x\xi[\eta]$.

# See also
 [`DξExp`](@ref), [`jacobiField`](@ref)
"""
DxExp(M::mT,x::P,ξ::T,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector}= jacobiField(M,x,exp(M,x,ξ),1.0,η,βDexpx)
@doc doc"""
    DξExp(M,x,ξ,η)
computes $D_\xi\exp_x\xi[\eta]$.
Note that $\xi\in T_\xi(T_x\mathcal M) = T_x\mathcal M$ is still a tangent vector.

# See also
 [`DxExp`](@ref), [`jacobiField`](@ref)
"""
DξExp(M::mT,x::P,ξ::T,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector} = jacobiField(M,x,exp(M,x,ξ),1.0,η,βDexpξ)
@doc doc"""
    DxLog(M,x,y,η)
computes $D_xlog_xy[\eta]$.

# See also
 [`DyLog`](@ref), [`jacobiField`](@ref)
"""
DxLog(M::mT,x::P,y::P,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector} = jacobiField(M,x,y,0.0,η,βDlogx)
@doc doc"""
    DyLog(M,x,y,η)
computes $D_ylog_xy[\eta]$.

# See also
 [`DxLog`](@ref), [`jacobiField`](@ref)
"""
DyLog(M::mT,x::P,y::P,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector} = jacobiField(M,y,x,1.0,η,βDlogy)

@doc doc"""
    ν = DforwardLogs(M,x,ξ)

compute the differenital of [`forwardLogs`](@ref) $F$ on the [`Power`](@ref) manifold
`M` at `x` and direction `ξ` ,
in the power manifold array, the differential of the function

```math
F_i(x) = \sum_{j\in\mathcal I_i} \log_{x_i} x_j$, \quad i \in \mathcal G,
```

where $\mathcal G$ is the set of indices of the [`Power`](@ref) manifold `M`
and $\mathcal I_i$ denotes the forward neighbors of $i$.

# Input
* `M`     – a [`Power`](@ref) manifold
* `x`     – a [`PowPoint`](@ref).
* `ξ`     – a [`PowTVector`](@ref).

# Ouput
* `ν` – resulting tangent vector in $T_x\mathcal N$ representing the differentials of the logs, where
  $\mathcal N$ is thw power manifold with the number of dimensions added to `size(x)`.
"""
function DforwardLogs(M::Power,x::PowPoint,ξ::PowTVector)::PowTVector
  sξ = size(ξ)
  R = CartesianIndices(sξ)
  d = length(sξ)
  maxInd = [last(R).I...]
  d2 = (d>1) ? ones(Int,d+1) + (d-1)*(1:(d+1) .== d+1 ) : d
  N = Power(M.manifold,(sξ...,d))
  ν = zeroTVector(N, repeat(x,inner=d2) )
  for i in R # iterate over all pixel
    for k in 1:d # for all direction combinations
      I = [i.I...] # array of index
      J = I .+ 1 .* (1:d .== k) #i + e_k is j
      if all( J .<= maxInd )
        # this is neighbor in range,
        j = CartesianIndex{d}(J...) # neigbbor index as Cartesian Index
        # collects two, namely in kth direction since xi appears as base and arg
        ν[i,k] = DxLog(M.manifold,x[i],x[j],ξ[i]) + DyLog(M.manifold,x[i],x[j],ξ[j])
      end
    end # directions
  end # i in R
  return ν
end
