module ManoptJuMPManifoldsExt

using Manopt
using ManifoldsBase
using Manifolds
using LinearAlgebra
using JuMP: JuMP
const MOI = JuMP.MOI
# the order of extensions is as such that
# ManoptJuMPExt is loaded after Manopt & JuMP but before this one.
const MJE = Base.get_extension(Manopt, :ManoptJuMPExt)
#
#
# define further conversions between certain Manifolds point/vector types
# and its vectorized representations for JuMP.
#
# Since Manopt & JuMP are already loaded we can asumme that the
# `Manopt.JuMPManifoldPointShape` and `Manopt.JuMPTangentVectorShape` are defined.
#
# #TODO: 1. check that we can use the types
# #TODO: 2. implement at least the hyperbolic conversions.

# TODO: These are just proof of concept functions to extend conversion to further types.
# vector -> point
function MJE._reshape_vector!(
        v::Vector{T},
        p::Manifolds.HyperboloidPoint,
        ::MJE.ManifoldPointShape{M, Manifolds.HyperboloidPoint},
    ) where {T, M <: ManifoldsBase.AbstractManifold}
    v .= p.value
    return v
end
function JuMP.reshape_vector(
        v::Vector{T}, shape::MJE.ManifoldPointShape{M, Manifolds.HyperboloidPoint}
    ) where {T, M <: ManifoldsBase.AbstractManifold}
    return HyperboloidPoint(v)
end
# point -> vector
function JuMP.vectorize(
        p::Manifolds.HyperboloidPoint, ::MJE.ManifoldPointShape{M, Manifolds.HyperboloidPoint}
    ) where {M <: ManifoldsBase.AbstractManifold}
    return p.value # is a vector already
end

function MJE._shape(m::M, ::Manifolds.HyperboloidPoint) where {M}
    return MJE.ManifoldPointShape{M, Manifolds.HyperboloidPoint}(m)
end

end # module ManoptJuMPManifoldsExt
