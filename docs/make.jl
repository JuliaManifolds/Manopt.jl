#!/usr/bin/env julia
#
#

#
# (a) if docs is not the current active environment, switch to it
# (from https://github.com/JuliaIO/HDF5.jl/pull/1020/) 
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.develop(PackageSpec(; path=(@__DIR__) * "/../"))
    Pkg.resolve()
    Pkg.instantiate()
end

# (b) Did someone say render? Then we render!
if "--quarto" ∈ ARGS
    using CondaPkg
    CondaPkg.withenv() do
        @info "Rendering Quarto"
        tutorials_folder = (@__DIR__) * "/../tutorials"
        # instantiate the tutorials environment if necessary
        Pkg.activate(tutorials_folder)
        Pkg.resolve()
        Pkg.instantiate()
        Pkg.build("IJulia") # build IJulia to the right version.
        Pkg.activate(@__DIR__) # but return to the docs one before
        run(`quarto render $(tutorials_folder)`)
    end
end

# (c) load necessary packages for the docs
using Documenter
using DocumenterCitations
using LineSearches, LRUCache, Manopt, Manifolds, Plots

# (d) add contributing.md to docs
generated_path = joinpath(@__DIR__, "src")
base_url = "https://github.com/JuliaManifolds/Manopt.jl/blob/master/"
isdir(generated_path) || mkdir(generated_path)
open(joinpath(generated_path, "contributing.md"), "w") do io
    # Point to source license file
    println(
        io,
        """
        ```@meta
        EditURL = "$(base_url)CONTRIBUTING.md"
        ```
        """,
    )
    # Write the contents out below the meta block
    for line in eachline(joinpath(dirname(@__DIR__), "CONTRIBUTING.md"))
        println(io, line)
    end
end

# (e) ...finally! make docs
bib = CitationBibliography(joinpath(@__DIR__, "src", "references.bib"); style=:alpha)
makedocs(
    bib;
    format=Documenter.HTML(;
        mathengine=MathJax3(), prettyurls=get(ENV, "CI", nothing) == "true"
    ),
    modules=[
        Manopt,
        if isdefined(Base, :get_extension)
            Base.get_extension(Manopt, :ManoptLineSearchesExt)
        else
            Manopt.ManoptLineSearchesExt
        end,
        if isdefined(Base, :get_extension)
            Base.get_extension(Manopt, :ManoptLRUCacheExt)
        else
            Manopt.ManoptLRUCacheExt
        end,
        if isdefined(Base, :get_extension)
            Base.get_extension(Manopt, :ManoptManifoldsExt)
        else
            Manopt.ManoptManifoldsExt
        end,
        if isdefined(Base, :get_extension)
            Base.get_extension(Manopt, :ManoptPlotsExt)
        else
            Manopt.ManoptPlotsExt
        end,
    ],
    authors="Ronny Bergmann and contributors.",
    sitename="Manopt.jl",
    strict=[
        :doctest,
        :linkcheck,
        :parse_error,
        :example_block,
        :autodocs_block,
        :cross_references,
        :docs_block,
        :eval_block,
        :example_block,
        :footnote,
        :meta_block,
        :missing_docs,
        :setup_block,
    ],
    pages=[
        "Home" => "index.md",
        "About" => "about.md",
        "How to..." => [
            "Get started: Optimize!" => "tutorials/Optimize!.md",
            "Speedup using Inplace computations" => "tutorials/InplaceGradient.md",
            "Use Automatic Differentiation" => "tutorials/AutomaticDifferentiation.md",
            "Define Objectives in the Embedding" => "tutorials/EmbeddingObjectives.md",
            "Count and use a Cache" => "tutorials/CountAndCache.md",
            "Print Debug Output" => "tutorials/HowToDebug.md",
            "Record values" => "tutorials/HowToRecord.md",
            "Implement a Solver" => "tutorials/ImplementASolver.md",
            "Do Constrained Optimization" => "tutorials/ConstrainedOptimization.md",
            "Do Geodesic Regression" => "tutorials/GeodesicRegression.md",
        ],
        "Solvers" => [
            "Introduction" => "solvers/index.md",
            "Adaptive Regularization with Cubics" => "solvers/adaptive-regularization-with-cubics.md",
            "Alternating Gradient Descent" => "solvers/alternating_gradient_descent.md",
            "Augmented Lagrangian Method" => "solvers/augmented_Lagrangian_method.md",
            "Chambolle-Pock" => "solvers/ChambollePock.md",
            "Conjugate gradient descent" => "solvers/conjugate_gradient_descent.md",
            "Cyclic Proximal Point" => "solvers/cyclic_proximal_point.md",
            "Difference of Convex" => "solvers/difference_of_convex.md",
            "Douglas–Rachford" => "solvers/DouglasRachford.md",
            "Exact Penalty Method" => "solvers/exact_penalty_method.md",
            "Frank-Wolfe" => "solvers/FrankWolfe.md",
            "Gradient Descent" => "solvers/gradient_descent.md",
            "Levenberg–Marquardt" => "solvers/LevenbergMarquardt.md",
            "Nelder–Mead" => "solvers/NelderMead.md",
            "Particle Swarm Optimization" => "solvers/particle_swarm.md",
            "Primal-dual Riemannian semismooth Newton" => "solvers/primal_dual_semismooth_Newton.md",
            "Quasi-Newton" => "solvers/quasi_Newton.md",
            "Stochastic Gradient Descent" => "solvers/stochastic_gradient_descent.md",
            "Subgradient method" => "solvers/subgradient.md",
            "Steihaug-Toint TCG Method" => "solvers/truncated_conjugate_gradient_descent.md",
            "Trust-Regions Solver" => "solvers/trust_regions.md",
        ],
        "Plans" => [
            "Specify a Solver" => "plans/index.md",
            "Problem" => "plans/problem.md",
            "Objective" => "plans/objective.md",
            "Solver State" => "plans/state.md",
            "Stepsize" => "plans/stepsize.md",
            "Stopping Criteria" => "plans/stopping_criteria.md",
            "Debug Output" => "plans/debug.md",
            "Recording values" => "plans/record.md",
        ],
        "Functions" => [
            "Introduction" => "functions/index.md",
            "Bézier curves" => "functions/bezier.md",
            "Cost functions" => "functions/costs.md",
            "Differentials" => "functions/differentials.md",
            "Adjoint Differentials" => "functions/adjoint_differentials.md",
            "Gradients" => "functions/gradients.md",
            "Proximal Maps" => "functions/proximal_maps.md",
            "Specific Manifold Functions" => "functions/manifold.md",
        ],
        "Helpers" => [
            "Checks" => "helpers/checks.md",
            "Data" => "helpers/data.md",
            "Error Measures" => "helpers/errorMeasures.md",
            "Exports" => "helpers/exports.md",
        ],
        "Contributing to Manopt.jl" => "contributing.md",
        "Extensions" => "extensions.md",
        "Notation" => "notation.md",
        "References" => "references.md",
    ],
)
deploydocs(; repo="github.com/JuliaManifolds/Manopt.jl", push_preview=true)
#back to main env
Pkg.activate()
