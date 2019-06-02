#
#      MatrixManifold -- a matrix manifold,
#       i.e. the values/points on the manifold are matrices
#		this is here implemented as traits, such that every manifold can
#		“gain” this feature by setting @traitimlp IsMatrixM{}
#		for the manifold, its point and the tangent vector
#
# Manopt.jl, R. Bergmann, 2018-06-26
export IsEmbeddedM, IsEmbeddedP, IsEmbeddedV
"""
    IsEmbeddedM{X}
An abstract [`Manifold`](@ref) that is embedded in some Euclidean space.
These manifolds may have projections and converters for gradient and Hessian.
"""
@traitdef IsEmbeddedM{X}
"""
    IsEmbeddedP{X}
An abstract [`MPoint`](@ref) belonging to an embedded manifold.
"""
@traitdef IsEmbeddedP{X}
"""
    IsEmbeddedV{X}
An abstract [`TVector`](@ref) belonging to an embedded manifold.
"""
@traitdef IsEmbeddedV{X}
