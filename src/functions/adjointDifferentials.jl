export AdjDxGeo, AdjDyGeo, AdjDxExp, AdjDξExp, AdjDxLog, AdjDyLog
export AdjDforwardLogs
@doc doc"""
    AdjDxGeo(M,x,y,t,η)
computes the adjoint of $D_xg(t;x,y)[\eta]$.

# See also
 [`DxGeo`](@ref), [`adjointJacobiField`](@ref)
"""
AdjDxGeo(M::mT,x::P,y::P,t::Float64,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector}= adjointJacobiField(M,x,y,t,η,βDgx)
@doc doc"""
    AdjDyGeo(M,x,y,t,η)
computes the adjoint of $D_yg(t;x,y)[\eta]$.

# See also
 [`DyGeo`](@ref), [`adjointJacobiField`](@ref)
"""
AdjDyGeo(M::mT,x::P,y::P,t::Float64,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector} = adjointJacobiField(M,y,x,1-t,η,βDgx)
@doc doc"""
    AdjDxExp(M,x,ξ,η)
computes the adjoint of $D_x\exp_x\xi[\eta]$.

# See also
 [`DxExp`](@ref), [`adjointJacobiField`](@ref)
"""
AdjDxExp(M::mT,x::P,ξ::T,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector} = adjointJacobiField(M,x,exp(M,x,ξ),1.,η,βDexpx)
@doc doc"""
    AdjDξExp(M,x,ξ,η)
computes the adjoint of $D_\xi\exp_x\xi[\eta]$.
Note that $\xi\in T_\xi(T_x\mathcal M) = T_x\mathcal M$ is still a tangent vector.

# See also
 [`DξExp`](@ref), [`adjointJacobiField`](@ref)
"""
AdjDξExp(M::mT,x::P,ξ::T,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector} = adjointJacobiField(M,x,exp(M,x,ξ),1.0,η,βDexpξ)
@doc doc"""
    AdjDxLog(M,x,y,η)
computes the adjoint of $D_xlog_xy[\eta]$.

# See also
 [`DxLog`](@ref), [`adjointJacobiField`](@ref)
"""
AdjDxLog(M::mT,x::P,y::P,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector} = adjointJacobiField(M,x,y,0.,η,βDlogx)
@doc doc"""
    AdjDyLog(M,x,y,η)
computes the adjoint of $D_ylog_xy[\eta]$.

# See also
 [`DyLog`](@ref), [`adjointJacobiField`](@ref)
"""
AdjDyLog(M::mT,x::P,y::P,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector} = adjointJacobiField(M,x,y,1.,η,βDlogy)
@doc doc"""
    ξ = AdjDforwardLogs(M,x,ν)

compute the adjoibnt differential of [`forwardLogs`](@ref) $F$ orrucirng,
in the power manifold array, the differential of the function

$F_i(x) = \sum_{j\in\mathcal I_i} \log_{x_i} x_j$

where $i$ runs over all indices of the [`Power`](@ref) manifold `M` and $\mathcal I_i$
denotes the forward neighbors of $i$
Let $n$ be the number dimensions of the [`Power`](@ref) manifold (i.e. `length(size(x)`)).
Then the input tangent vector lies on the manifold $\mathcal M' = \mathcal M^n$.

# Input
* `M`     – a [`Power`](@ref) manifold
* `x`     – a [`PowPoint`](@ref).
* `ν`     – a [`PowTVector`](@ref) from $T_X\mathcal M'$, where
  $X = (x,...,x)\in\mathcal M'$ is an $n$-fold copy of $x$ where $\mathcal N (x,...,x)N.

# Ouput
* ξ – resulting tangent vector in $T_x\mathcal M$ representing the adjoint
  differentials of the logs.
"""
function AdjDforwardLogs(M::Power,x::PowPoint,ν::PowTVector)::PowTVector
  sX = size(x)
  R = CartesianIndices(sX)
  d = length(sX)
  maxInd = [last(R).I...] # maxInd as Array
  N = Power(M.manifold,(sX...,d))
  ξ = zeroTVector(M,x)
  for i in R # iterate over all pixel
    for k in 1:d # for all direction combinations
        I = [i.I...] # array of index
        J = I .+ 1 .* (1:d .== k) #i + e_k is j
        if all( J .<= maxInd ) # is this neighbor in range?
            j = CartesianIndex{d}(J...) # neigbbor index as Cartesian Index
            ξ[i] += AdjDxLog(M.manifold,x[i],x[j],ν[i,k])
            ξ[j] += AdjDyLog(M.manifold,x[i],x[j],ν[i,k])
        end
    end # directions
  end # i in R
  return ξ
end
