export AdjDxGeo, AdjDyGeo, AdjDxexp, AdjDξexp, AdjDxLog, AdjDylog
AdjDxgeo{mT <: Manifold, P <: MPoint, T<: TVector}(M::mT,x::P,y::P,t::Number,η::T) = AdjointJacobiField(M,x,y,t,η,βDgx)
AdjDygeo{mT <: Manifold, P <: MPoint, T<: TVector}(M::mT,x::P,y::P,t::Number,η::T) = AdjointJacobiField(M,x,y,1-t,η,βDgy)
AdjDxexp{mT <: Manifold, P <: MPoint, T<: TVector}(M::mT,x::P,ξ::P,t::Number,η::T) = AdjointJacobiField(M,x,exp(M,x,ξ),1,η,βDexpx)
AdjDξexp{mT <: Manifold, P <: MPoint, T<: TVector}(M::mT,x::P,y::P,t::Number,η::T) = AdjointJacobiField(M,x,exp(M,x,ξ),1,η,βDexpξ)
AdjDxlog{mT <: Manifold, P <: MPoint, T<: TVector}(M::mT,x::P,y::P,t::Number,η::T) = AdjointJacobiField(M,x,y,0,η,βDlogx)
AdjDylog{mT <: Manifold, P <: MPoint, T<: TVector}(M::mT,x::P,y::P,t::Number,η::T) = AdjointJacobiField(M,y,x,1,η,βDylogy)
