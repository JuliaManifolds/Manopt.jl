export DxGeo, DyGeo, DxExp,DξExp, DxLog, DyLog
@doc doc"""
    DxGeo(M,x,y,t,η)
computes $D_xg(t;x,y)[\eta]$.

*See also:* [`DyGeo`](@ref), [`jacobiField`](@ref)
"""
DxGeo(M::mT,x::P,y::P,t::Number,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector} = jacobiField(M,x,y,t,η,βDgx)
@doc doc"""
    DyGeo(M,x,y,t,η)
computes $D_yg(t;x,y)[\eta]$.

*See also:* [`DxGeo`](@ref), [`jacobiField`](@ref)
"""
DyGeo(M::mT,x::P,y::P,t::Number,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector} = jacobiField(M,x,y,1.0-t,η,βDgy)
@doc doc"""
    DxExp(M,x,ξ,η)
computes $D_x\exp_x\xi[\eta]$.

*See also:* [`DξExp`](@ref), [`jacobiField`](@ref)
"""
DxExp(M::mT,x::P,ξ::T,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector}= jacobiField(M,x,exp(M,x,ξ),1.0,η,βDexpx)
@doc doc"""
    DξExp(M,x,ξ,η)
computes $D_\xi\exp_x\xi[\eta]$.
Note that $\xi\in T_\xi(T_x\mathcal M) = T_x\mathcal M$ is still a tangent vector.

*See also:* [`DxExp`](@ref), [`jacobiField`](@ref)
"""
DξExp(M::mT,x::P,ξ::T,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector} = jacobiField(M,x,exp(M,x,ξ),1.0,η,βDexpξ)
@doc doc"""
    DxLog(M,x,y,η)
computes $D_xlog_xy[\eta]$.

*See also:* [`DyLog`](@ref), [`jacobiField`](@ref)
"""
DxLog(M::mT,x::P,y::P,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector} = jacobiField(M,x,y,0.0,η,βDlogx)
@doc doc"""
    DyLog(M,x,y,η)
computes $D_ylog_xy[\eta]$.

*See also:* [`DxLog`](@ref), [`jacobiField`](@ref)
"""
DyLog(M::mT,x::P,y::P,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector} = jacobiField(M,y,x,1.0,η,βDlogy)
