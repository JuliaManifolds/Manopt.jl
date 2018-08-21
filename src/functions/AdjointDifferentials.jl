export AdjDxGeo, AdjDyGeo, AdjDxexp, AdjDξexp, AdjDxLog, AdjDylog
AdjDxgeo(M::mT,x::P,y::P,t::Number,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector}= AdjointJacobiField(M,x,y,t,η,βDgx)
AdjDygeo(M::mT,x::P,y::P,t::Number,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector} = AdjointJacobiField(M,x,y,1-t,η,βDgy)
AdjDxexp(M::mT,x::P,ξ::P,t::Number,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector} = AdjointJacobiField(M,x,exp(M,x,ξ),1,η,βDexpx)
AdjDξexp(M::mT,x::P,y::P,t::Number,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector} = AdjointJacobiField(M,x,exp(M,x,ξ),1,η,βDexpξ)
AdjDxlog(M::mT,x::P,y::P,t::Number,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector} = AdjointJacobiField(M,x,y,0,η,βDlogx)
AdjDylog(M::mT,x::P,y::P,t::Number,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector} = AdjointJacobiField(M,y,x,1,η,βDylogy)
