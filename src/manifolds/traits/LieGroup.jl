#
# A Lee Group as a trait to manifolds
#
#
# Manopt.jl, R. Bergmann, 2018-07-07
export IsLieGroupM, IsLieGroupP, IsLieGroupV, ⊗

"""
    IsLieGroupM{X}
Indicates that `X` is a [`Manifold`](@ref) with a Lie group structure. This also
introdcues a group operation [`⊗`](@ref) of two [`MPoint`](@ref)s of `X`.
"""
@traitdef IsLieGroupM{X}
"""
    IsLieGroupP{X}
An abstract [`MPoint`](@ref) belonging to a Lie group manifold.
"""
@traitdef IsLieGroupP{X}
"""
    IsLieGroupV{X}
An abstract [`TVector`](@ref) belonging to a Lie group manifold.
"""
@traitdef IsLieGroupV{X}
@doc doc"""
    ⊗(x,y)

the binary operator `x ⊗ y` represents the Lie group action for a Lie group
[`Manifold`](@ref), see [`IsLieGroupM`](@ref). 
"""
⊗(x::P,y::P) where {P <: MPoint}= LieGroupOp(x,y)
@traitfn function LieGroupOp(x::P,y::P) where {P <: MPoint; !IsLieGroupP{P}}
    sig1 = string( typeof(x) )
    throw( ErrorException("The manifold point of type $sig1 does not belong to a Lie Group") );
end
