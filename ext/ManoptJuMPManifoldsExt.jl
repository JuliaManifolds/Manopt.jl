module ManoptJuMPManifoldsExt

using Manopt
using ManifoldsBase
using Manifolds
using LinearAlgebra
using JuMP: JuMP
const MOI = JuMP.MOI
# Finally found a hack to bring functions here as well.
const MJE = Base.get_extension(Manopt, :ManoptJuMPExt)
#
#
# define further conversions between certain Manifolds point/vector types
# and its vectorized representations for JuMP.
#
# Since Manopt & JuMP are already loaded we can asume that the
# `Manopt.JuMPManifoldPointShape` and `Manopt.JuMPTangentVectorShape` are defined.
#
# #TODO: 1. check that we can use the types
# #TODO: 2. implement at least the hyperbolic conversions.

# TODO: These are just proof of concept functions to extend conversion to further types.
# vector -> point
function MJE._reshape_vector!(
    v::Vector{T},
    p::Manifolds.HyperboloidPoint,
    ::MJE.ManifoldPointShape{M,Manifolds.HyperboloidPoint},
) where {T,M<:ManifoldsBase.AbstractManifold}
    v .= p.value
    return v
end
function JuMP.reshape_vector(
    v::Vector{T}, shape::MJE.ManifoldPointShape{M,Manifolds.HyperboloidPoint}
) where {T,M<:ManifoldsBase.AbstractManifold}
    p = rand(shape.manifold, HyperboloidPoint)
    p.value .= v
    return p
end
# point -> vector
function JuMP.vectorize(
    p::Manifolds.HyperboloidPoint,
    shape::MJE.ManifoldPointShape{M,Manifolds.HyperboloidPoint},
) where {M<:ManifoldsBase.AbstractManifold}
    return p.value # is a vector already
end

end # module ManoptJuMPManifoldsExt
