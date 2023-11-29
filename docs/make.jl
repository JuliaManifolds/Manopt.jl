#!/usr/bin/env julia
#
#

if "--help" ∈ ARGS
    println(
        """
docs/make.jl

Render the `Manopt.jl` documenation with optinal arguments

Arguments
* `--exclude-docs` - exclude the tutorials from the menu of Documenter,
  this can be used if you do not have Quarto installed to still be able to render the docs
  locally on this machine. This option should not be set on CI.
* `--help`         - print this help and exit without rendering the documentation
* `--prettyurls`   – toggle the prettyurls part to true (which is otherwise only true on CI)
* `--quarto`       – run the Quarto notebooks from the `tutorials/` folder before generating the documentation
  this has to be run locally at least once for the `tutorials/*.md` files to exist that are included in
  the documentation (see `--exclude-tutorials`) for the alternative.
  If they are generated ones they are cached accordingly.
  Then you can spare time in the rendering by not passing this argument.
""",
    )
    exit(0)
end

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

tutorials_in_menu = true
if "--exclude-tutorials" ∈ ARGS
    @warn """
    You are excluding the tutorials from the Menu,
    which might be done if you can not render them locally.

    Remember that this should never be done on CI for the full documentation.
    """
    tutorials_in_menu = false
end

# (c) load necessary packages for the docs
using Documenter
using DocumenterCitations
using JuMP, LineSearches, LRUCache, Manopt, Manifolds, Plots

# (d) add contributing.md to docs
generated_path = joinpath(@__DIR__, "src")
base_url = "https://github.com/JuliaManifolds/Manopt.jl/blob/master/"
isdir(generated_path) || mkdir(generated_path)
for (md_file, doc_file) in
    [("CONTRIBUTING.md", "contributing.md"), ("Changelog.md", "changelog.md")]
    open(joinpath(generated_path, doc_file), "w") do io
        # Point to source license file
        println(
            io,
            """
            ```@meta
            EditURL = "$(base_url)$(md_file)"
            ```
            """,
        )
        # Write the contents out below the meta block
        for line in eachline(joinpath(dirname(@__DIR__), md_file))
            println(io, line)
        end
    end
end

## Build titorials menu
tutorials_menu =
    "How to..." => [
        "🏔️ Get started: optimize." => "tutorials/Optimize.md",
        "Speedup using in-place computations" => "tutorials/InplaceGradient.md",
        "Use automatic differentiation" => "tutorials/AutomaticDifferentiation.md",
        "Define objectives in the embedding" => "tutorials/EmbeddingObjectives.md",
        "Count and use a cache" => "tutorials/CountAndCache.md",
        "Print debug output" => "tutorials/HowToDebug.md",
        "Record values" => "tutorials/HowToRecord.md",
        "Implement a solver" => "tutorials/ImplementASolver.md",
        "Optimize on your own manifold" => "tutorials/ImplementOwnManifold.md",
        "Do constrained optimization" => "tutorials/ConstrainedOptimization.md",
        "Do geodesic regression" => "tutorials/GeodesicRegression.md",
    ]
# (e) ...finally! make docs
bib = CitationBibliography(joinpath(@__DIR__, "src", "references.bib"); style=:alpha)
makedocs(;
    format=Documenter.HTML(;
        prettyurls=(get(ENV, "CI", nothing) == "true") || ("--prettyurls" ∈ ARGS),
        assets=["assets/favicon.ico", "assets/citations.css"],
    ),
    modules=[
        Manopt,
        if isdefined(Base, :get_extension)
            Base.get_extension(Manopt, :ManoptJuMPExt)
        else
            Manopt.ManoptJuMPExt
        end,
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
    pages=[
        "Home" => "index.md",
        "About" => "about.md",
        (tutorials_in_menu ? [tutorials_menu] : [])...,
        "Solvers" => [
            "Introduction" => "solvers/index.md",
            "Adaptive Regularization with Cubics" => "solvers/adaptive-regularization-with-cubics.md",
            "Alternating Gradient Descent" => "solvers/alternating_gradient_descent.md",
            "Augmented Lagrangian Method" => "solvers/augmented_Lagrangian_method.md",
            "Chambolle-Pock" => "solvers/ChambollePock.md",
            "Conjugate gradient descent" => "solvers/conjugate_gradient_descent.md",
            "Cyclic Proximal Point" => "solvers/cyclic_proximal_point.md",
            "Difference of Convex" => "solvers/difference_of_convex.md",
            "Douglas—Rachford" => "solvers/DouglasRachford.md",
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
        "Helpers" => ["Checks" => "helpers/checks.md", "Exports" => "helpers/exports.md"],
        "Contributing to Manopt.jl" => "contributing.md",
        "Extensions" => "extensions.md",
        "Notation" => "notation.md",
        "Changelog" => "changelog.md",
        "References" => "references.md",
    ],
    plugins=[bib],
)
deploydocs(; repo="github.com/JuliaManifolds/Manopt.jl", push_preview=true)
#back to main env
Pkg.activate()
