# Extensions

`Manopt.jl` has several extensions that provide additional functionality as soon as certain packages are loaded.

* [`JuMP.jl`](https://jump.dev) see [the separate page](JuMP.md) how to use `Manopt.jl`s solver within the JuMP framework
* [`LineSearches.jl`](https://github.com/JuliaNLSolvers/LineSearches.jl) allows to use line search algorithms implemented in [LineSearches.jl](https://github.com/JuliaOptimizers/LineSearches.jl), see [the separate page](LineSearches.md) for more details.
* [`LURCache.jl`](https://github.com/JuliaCollections/LRUCache.jl) allows to use the extended caching capabilities
* [`Manifolds.jl`](@extref Manifolds :doc:`index`) see [the separate page](Manifolds.md) for more details.
* [`RecursiveArrayTools.jl`](https://docs.sciml.ai/RecursiveArrayTools/stable/) allows to use the [`alternating_gradient_descent`](@ref) solver in a product manifold.
* [`RipQP.jl`](https://jso.dev/RipQP.jl/stable/) and [`QuadraticModels.jl`](https://jso.dev/QuadraticModels.jl/stable/) together activate a closed form subsolvers for the [`convex_bundle_method`](@ref) and the [`proximal_bundle_method`](@ref) solvers.