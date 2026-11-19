module ManoptManifoldsExt

using ManifoldsBase: exp, log
using Manopt
using Manopt: get_stepsize_bound
import Manopt: directional_max_stepsize, max_stepsize, Rn
import ManifoldsBase: mid_point, mid_point!

using Manifolds

Rn(::Val{:Manifolds}, args...; kwargs...) = Euclidean(args...; kwargs...)

include("manifold_functions.jl")
include("ChambollePockManifolds.jl")
include("test_examples.jl")
end
