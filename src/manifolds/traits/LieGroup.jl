#
# A Lee Group as a trait to manifolds
#
#
# Manopt.jl, R. Bergmann, 2018-07-07
@traitdef IsLieGroupM{X}
@traitdef IsLieGroupP{X}
@traitdef IsLieGroupV{X}

# introduces a shorthand for the group operation
@traitfn ⊗{T <: MPoint; IsMatrixP{T}}(x::T,y::T) = LieGroupOp(x,y)
@traitfn function ⊗{T <: MPoint; !IsMatrixP{T}}(x::T,y::T)
        sig1 = string( typeof(x) )
        throw( ErrorException("The manifold point of type $sig1 does not belong to a Lie Group") );
end
@traitfn function LieGroupOp{T <: MPoint; !IsMatrixP{T}}(x::T,y::T)
    sig1 = string( typeof(x) )
    throw( ErrorException("The manifold point of type $sig1 does not belong to a Lie Group") );
end
