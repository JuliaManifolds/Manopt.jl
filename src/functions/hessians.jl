
approximateHessianForwardDifferences{GP <: GradientProblem, P <: MPoint, T <: TVector}(p::GradientProblem,x::P,ξ::T,stepSize::Float64=2^(-14))
    if norm(M,x,ξ) < eps(Float64)
