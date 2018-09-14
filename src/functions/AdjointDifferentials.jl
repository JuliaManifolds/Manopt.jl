export AdjDxGeo, AdjDyGeo, AdjDxexp, AdjDξexp, AdjDxLog, AdjDylog
AdjDxGeo(M::mT,x::P,y::P,t::Number,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector}= AdjointJacobiField(M,x,y,t,η,βDgx)
AdjDyGeo(M::mT,x::P,y::P,t::Number,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector} = AdjointJacobiField(M,x,y,1-t,η,βDgy)
AdjDxExp(M::mT,x::P,ξ::P,t::Number,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector} = AdjointJacobiField(M,x,exp(M,x,ξ),1,η,βDexpx)
AdjDξExp(M::mT,x::P,y::P,t::Number,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector} = AdjointJacobiField(M,x,exp(M,x,ξ),1,η,βDexpξ)
AdjDxLog(M::mT,x::P,y::P,t::Number,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector} = AdjointJacobiField(M,x,y,0,η,βDlogx)
AdjDyLog(M::mT,x::P,y::P,t::Number,η::T) where {mT <: Manifold, P <: MPoint, T<: TVector} = AdjointJacobiField(M,y,x,1,η,βDylogy)
