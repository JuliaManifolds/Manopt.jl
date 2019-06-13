#
#
#
export
@doc doc"""

"""
function truncatedConjugateGradient(hessprob::HessianProblem, x::MP,
    grad::Function, eta::MTVec, Delta::Float64,
    options::TruncatedConjugateGradientOptions) where {MP <: MPoint, MTVec <: MTVector}
    
end
