export AdjDxGeo, AdjDyGeo, AdjDxExp, AdjDξExp, AdjDxLog, AdjDyLog
@doc doc"""
    AdjDxGeo(M,x,y,t,η)
computes the adjoint of $D_xg(t;x,y)[\eta]$.

*See also:* [`DxGeo`](@ref), [`adjointJacobiField`](@ref)
"""
AdjDxGeo(M::mT,x::P,y::P,t::Number,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector}= adjointJacobiField(M,x,y,t,η,βDgx)
@doc doc"""
    AdjDyGeo(M,x,y,t,η)
computes the adjoint of $D_yg(t;x,y)[\eta]$.

*See also:* [`DyGeo`](@ref), [`adjointJacobiField`](@ref)
"""
AdjDyGeo(M::mT,x::P,y::P,t::Number,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector} = adjointJacobiField(M,x,y,1-t,η,βDgy)
@doc doc"""
    AdjDxExp(M,x,ξ,η)
computes the adjoint of $D_x\exp_x\xi[\eta]$.

*See also:* [`DxExp`](@ref), [`adjointJacobiField`](@ref)
"""
AdjDxExp(M::mT,x::P,ξ::P,t::Number,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector} = adjointJacobiField(M,x,exp(M,x,ξ),1,η,βDexpx)
@doc doc"""
    AdjDξExp(M,x,ξ,η)
computes the adjoint of $D_\xi\exp_x\xi[\eta]$.
Note that $\xi\in T_\xi(T_x\mathcal M) = T_x\mathcal M$ is still a tangent vector.

*See also:* [`DξExp`](@ref), [`adjointJacobiField`](@ref)
"""
AdjDξExp(M::mT,x::P,y::P,t::Number,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector} = adjointJacobiField(M,x,exp(M,x,ξ),1,η,βDexpξ)
@doc doc"""
    AdjDxLog(M,x,y,η)
computes the adjoint of $D_xlog_xy[\eta]$.

*See also:* [`DxLog`](@ref), [`adjointJacobiField`](@ref)
"""
AdjDxLog(M::mT,x::P,y::P,t::Number,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector} = adjointJacobiField(M,x,y,0,η,βDlogx)
@doc doc"""
    AdjDyLog(M,x,y,η)
computes the adjoint of $D_ylog_xy[\eta]$.

*See also:* [`DyLog`](@ref), [`adjointJacobiField`](@ref)
"""
AdjDyLog(M::mT,x::P,y::P,t::Number,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector} = adjointJacobiField(M,y,x,1,η,βDylogy)
