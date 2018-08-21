export DxGeo, DyGeo, Dxexp,Dξexp, DxLog, Dylog
Markdown.doc"""
    Dxgeo(M,x,y,t,η)
computes $D_xg(t;x,y)[\eta]$.

*See also:* [`Dygeo`](@ref), [`jacobiField`](@ref)
"""
Dxgeo(M::mT,x::P,y::P,t::Number,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector} = jacobiField(M,x,y,t,η,βDgx)
Markdown.doc"""
    Dygeo(M,x,y,t,η)
computes $D_yg(t;x,y)[\eta]$.

*See also:* [`Dxgeo`](@ref), [`jacobiField`](@ref)
"""
Dygeo(M::mT,x::P,y::P,t::Number,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector} = jacobiField(M,x,y,1-t,η,βDgy)
Markdown.doc"""
    Dxexp(M,x,ξ,η)
computes $D_x\exp_x\xi[\eta]$.

*See also:* [`Dξexp`](@ref), [`jacobiField`](@ref)
"""
Dxexp(M::mT,x::P,ξ::T,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector}= jacobiField(M,x,exp(M,x,ξ),1,η,βDexpx)
Markdown.doc"""
    Dξexp(M,x,ξ,η)
computes $D_\xi\exp_x\xi[\eta]$.
Note that $\xi\in T_\xi(T_x\mathcal M) = T_x\mathcal M$ is still a tangent vector.

*See also:* [`Dxexp`](@ref), [`jacobiField`](@ref)
"""
Dξexp(M::mT,x::P,ξ::T,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector} = jacobiField(M,x,exp(M,x,ξ),1,η,βDexpξ)
Markdown.doc"""
    DxLog(M,x,y,η)
computes $D_xlog_xy[\eta]$.

*See also:* [`Dylog`](@ref), [`jacobiField`](@ref)
"""
Dxlog(M::mT,x::P,y::P,t::Number,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector} = jacobiField(M,x,y,0,η,βDlogx)
Markdown.doc"""
    DyLog(M,x,y,η)
computes $D_ylog_xy[\eta]$.

*See also:* [`Dxlog`](@ref), [`jacobiField`](@ref)
"""
Dylog(M::mT,x::P,y::P,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector} = jacobiField(M,y,x,1,η,βDylogy)
