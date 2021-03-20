# see issue #64
# test for finding mean on a SE2 manifold

using Test
using StaticArrays
using Manopt, Manifolds

##

@testset "Test novice use-case with SpecialEuclidean(2) and optimizers" begin

##

M = SpecialEuclidean(2)
n = 100

# convert point to coordinates
coords(p) = SA[p.parts[1][1], p.parts[1][2], acos(p.parts[2][1,1])]
function uncoords(p)
    α = p[3] 
    return ProductRepr((SA[p[1], p[2]]), SA[cos(α) -sin(α); sin(α) cos(α)])
end


##

# some random points to make a tree from
data = [uncoords(@SVector randn(3)) for _ in 1:n]

# a test point
x = ProductRepr([0.0,0.0], [1 0; 0 1.0])

##

F(M, y) = sum(1 / (2 * n) * distance.(Ref(M), Ref(y), data) .^ 2)

# test objective function
F(M, x)

# automatically find gradients?
gradF(M, y) = sum(1 / n * grad_distance.(Ref(M), data, Ref(y)))


## Run gradient decent optimization

xMean = gradient_descent(M, F, gradF, data[1])

##
 
@warn "Rather standarize dispatch for ProductRepr when broadcasting?"
Base.size(::ProductRepr) = 1
Base.length(::ProductRepr) = 1
Base.iterate(v::ProductRepr, i=1) = (1 < i ? nothing : (v, i + 1))
# struggling to define a generic dispath for ndims on ProductRepr
Base.ndims(::Type{ProductRepr{Tuple{MVector{2, Float64}, MMatrix{2, 2, Float64, 4}}}} ) = 3
# Base.firstindex(v::ProductRepr) = 1
# Base.lastindex(v::ProductRepr) = 1
# Base.keys(v::ProductRepr) = Base.OneTo(length(v))
# Base.isempty(v::ProductRepr) = false


## Try gradient free optimizer


dim = manifold_dimension(M)
xMean = NelderMead(M, F, data[1:(dim+1)])

##


# # and example element from SpecialEuclidean(2)
# v = ProductRepr{Tuple{MVector{2, Float64}, MMatrix{2, 2, Float64, 4}}}(
#   ([0.0, 0.0], 
#   [0.0 0.0; 0.0 0.0])
# )


end


#
