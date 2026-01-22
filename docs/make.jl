#!/usr/bin/env julia
#
#

if "--help" ∈ ARGS
    println(
        """
        docs/make.jl

        Render the `Manopt.jl` documentation with optional arguments

        Arguments
        * `--exclude-tutorials` - exclude the tutorials from the menu of Documenter,
          This can be used if not all tutorials are rendered and you want to therefore exclude links
          to these, especially the corresponding menu. This option should not be set on CI.
          Locally this is also set if `--quarto` is not set and not all tutorials are rendered.
        * `--help`              - print this help and exit without rendering the documentation
        * `--prettyurls`        – toggle the pretty urls part to true, which is always set on CI
        * `--quarto`            – (re)run the Quarto notebooks from the `tutorials/` folder before
          generating the documentation. If they are generated once they are cached accordingly.
          Then you can spare time in the rendering by not passing this argument.
          If quarto is not run, some tutorials are generated as empty files, since they
          are referenced from within the documentation.
          These are currently `getstarted.md` and `ImplementOwnManifold.md`.
        """
    )
    exit(0)
end

run_quarto = "--quarto" in ARGS
run_on_CI = (get(ENV, "CI", nothing) == "true")
tutorials_in_menu = !("--exclude-tutorials" ∈ ARGS)
#
#
# (a) setup the tutorials menu – check whether all files exist
tutorials_menu =
    "How to..." => [
    "🏔️ Get started with Manopt.jl" => "tutorials/getstarted.md",
    "Speedup using in-place computations" => "tutorials/InplaceGradient.md",
    "Use automatic differentiation" => "tutorials/AutomaticDifferentiation.md",
    "Define objectives in the embedding" => "tutorials/EmbeddingObjectives.md",
    "Count and use a cache" => "tutorials/CountAndCache.md",
    "Print debug output" => "tutorials/HowToDebug.md",
    "Record values" => "tutorials/HowToRecord.md",
    "Implement a solver" => "tutorials/ImplementASolver.md",
    "Optimize on your own manifold" => "tutorials/ImplementOwnManifold.md",
    "Do constrained optimization" => "tutorials/ConstrainedOptimization.md",
]
# Check whether all tutorials are rendered, issue a warning if not (and quarto if not set)
all_tutorials_exist = true
for (name, file) in tutorials_menu.second
    fn = joinpath(@__DIR__, "src/", file)
    if !isfile(fn) || filesize(fn) == 0 # nonexistent or empty file
        global all_tutorials_exist = false
        if !run_quarto
            @warn "Tutorial $name does not exist at $fn."
            if (!isfile(fn)) && (
                    endswith(file, "getstarted.md") || endswith(file, "ImplementOwnManifold.md") || endswith(file, "HowToRecord.md")
                )
                @warn "Generating empty file, since this tutorial is linked to from the documentation."
                touch(fn)
            end
        end
    end
end
if !all_tutorials_exist && !run_quarto && !run_on_CI
    @warn """
        Not all tutorials exist. Run `make.jl --quarto` to generate them. For this run they are excluded from the menu.
    """
    tutorials_in_menu = false
end
if !tutorials_in_menu
    @warn """
    You are either explicitly or implicitly excluding the tutorials from the documentation.
    You will not be able to see their menu entries nor their rendered pages.
    """
    run_on_CI &&
        (@error "On CI, the tutorials have to be either rendered with Quarto or be cached.")
end
#
# (b) if docs is not the current active environment, switch to it
# (from https://github.com/JuliaIO/HDF5.jl/pull/1020/) 
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
end

# (b) If quarto is set, or we are on CI, run quarto
if run_quarto || run_on_CI
    @info "Rendering Quarto"
    tutorials_folder = (@__DIR__) * "/../tutorials"
    # instantiate the tutorials environment if necessary
    Pkg.activate(tutorials_folder)
    # For a breaking release -> also set the tutorials folder to the most recent version
    Pkg.instantiate()
    Pkg.activate(@__DIR__) # but return to the docs one before
    run(`quarto render $(tutorials_folder)`)
end

# (c) load necessary packages for the docs
using Documenter
using DocumenterCitations, DocumenterInterLinks
using JuMP, LineSearches, LRUCache, Manopt, Manifolds, Plots, RecursiveArrayTools
using RipQP, QuadraticModels

# (d) add contributing.md and changelog.md to the docs – and link to releases and issues

function add_links(line::String, url::String = "https://github.com/JuliaManifolds/Manopt.jl")
    # replace issues (#XXXX) -> ([#XXXX](url/issue/XXXX))
    while (m = match(r"\(\#([0-9]+)\)", line)) !== nothing
        id = m.captures[1]
        line = replace(line, m.match => "([#$id]($url/issues/$id))")
    end
    # replace ## [X.Y.Z] -> with a link to the release [X.Y.Z](url/releases/tag/vX.Y.Z)
    while (m = match(r"\#\# \[([0-9]+.[0-9]+.[0-9]+)\] (.*)", line)) !== nothing
        tag = m.captures[1]
        date = m.captures[2]
        line = replace(line, m.match => "## [$tag]($url/releases/tag/v$tag) ($date)")
    end
    return line
end

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
            println(io, add_links(line))
        end
    end
end

## Build tutorials menu
# (e) finally make docs
bib = CitationBibliography(joinpath(@__DIR__, "src", "references.bib"); style = :alpha)
links = InterLinks(
    "JuMP" => ("https://jump.dev/JuMP.jl/stable/"),
    "ManifoldDiff" => ("https://juliamanifolds.github.io/ManifoldDiff.jl/stable/"),
    "ManifoldsBase" => ("https://juliamanifolds.github.io/ManifoldsBase.jl/stable/"),
    "Manifolds" => ("https://juliamanifolds.github.io/Manifolds.jl/stable/"),
)
makedocs(;
    format = Documenter.HTML(;
        prettyurls = run_on_CI || ("--prettyurls" ∈ ARGS),
        assets = ["assets/favicon.ico", "assets/citations.css", "assets/link-icons.css"],
        size_threshold = 1100 * 2^10,      # raise slightly 200 to to 300 KiB
        size_threshold_warn = 900 * 2^10, # raise from 500 KiB to 1.1 MB (for search index)
        search_size_threshold_warn = 2000 * 2^10,
    ),
    modules = [
        Manopt,
        Base.get_extension(Manopt, :ManoptJuMPExt),
        Base.get_extension(Manopt, :ManoptLineSearchesExt),
        Base.get_extension(Manopt, :ManoptLRUCacheExt),
        Base.get_extension(Manopt, :ManoptManifoldsExt),
        Base.get_extension(Manopt, :ManoptRipQPQuadraticModelsExt),
    ],
    authors = "Ronny Bergmann <ronny.bergmann@ntnu.no> and contributors.",
    sitename = "Manopt.jl",
    pages = [
        "Home" => "index.md",
        "About" => "about.md",
        (tutorials_in_menu ? [tutorials_menu] : [])...,
        "Solvers" => [
            "List of Solvers" => "solvers/index.md",
            "Adaptive Regularization with Cubics" => "solvers/adaptive-regularization-with-cubics.md",
            "Alternating Gradient Descent" => "solvers/alternating_gradient_descent.md",
            "Augmented Lagrangian Method" => "solvers/augmented_Lagrangian_method.md",
            "Chambolle-Pock" => "solvers/ChambollePock.md",
            "CMA-ES" => "solvers/cma_es.md",
            "Conjugate gradient descent" => "solvers/conjugate_gradient_descent.md",
            "Conjugate Residual" => "solvers/conjugate_residual.md",
            "Convex bundle method" => "solvers/convex_bundle_method.md",
            "Cyclic Proximal Point" => "solvers/cyclic_proximal_point.md",
            "Difference of Convex" => "solvers/difference_of_convex.md",
            "Douglas—Rachford" => "solvers/DouglasRachford.md",
            "Exact Penalty Method" => "solvers/exact_penalty_method.md",
            "Frank-Wolfe" => "solvers/FrankWolfe.md",
            "Gradient Descent" => "solvers/gradient_descent.md",
            "Interior Point Newton" => "solvers/interior_point_Newton.md",
            "Levenberg–Marquardt" => "solvers/LevenbergMarquardt.md",
            "Mesh Adaptive Direct Search" => "solvers/mesh_adaptive_direct_search.md",
            "Nelder–Mead" => "solvers/NelderMead.md",
            "Particle Swarm Optimization" => "solvers/particle_swarm.md",
            "Primal-dual Riemannian semismooth Newton" => "solvers/primal_dual_semismooth_Newton.md",
            "Projected Gradient Method" => "solvers/projected_gradient_method.md",
            "Proximal bundle method" => "solvers/proximal_bundle_method.md",
            "Proximal Gradient Method" => "solvers/proximal_gradient_method.md",
            "Quasi-Newton" => "solvers/quasi_Newton.md",
            "Stochastic Gradient Descent" => "solvers/stochastic_gradient_descent.md",
            "Subgradient method" => "solvers/subgradient.md",
            "Steihaug-Toint TCG Method" => "solvers/truncated_conjugate_gradient_descent.md",
            "Trust-Regions Solver" => "solvers/trust_regions.md",
            "Vector Bundle Newton Method" => "solvers/vectorbundle_newton.md",
        ],
        "Plans" => [
            "Specify a Solver" => "plans/index.md",
            "Problem" => "plans/problem.md",
            "Objective" => [
                "Introduction" => "plans/objective.md",
                "Cost objectvies" => "plans/objectives/cost.md",
                "First-order objectives" => "plans/objectives/first_order.md",
                "Second-order objectives" => "plans/objectives/second_order.md",
                "Constrained objectives" => "plans/objectives/constrained.md",
                "Splitting-based objectives" => "plans/objectives/splitting_based.md",
                "Subproblem objectives" => "plans/objectives/sub.md",
                "Vectorial objectives" => "plans/objectives/vectorial.md",
                "Linear Systems" => "plans/objectives/linear_system.md",
                "Decorators for objectives" => "plans/objectives/decorated.md",
            ],
            "Solver State" => "plans/state.md",
            "Stepsize" => "plans/stepsize.md",
            "Stopping Criteria" => "plans/stopping_criteria.md",
            "Debug Output" => "plans/debug.md",
            "Recording values" => "plans/record.md",
        ],
        "Helpers" => [
            "Checks" => "helpers/checks.md",
            "Exports" => "helpers/exports.md",
            "Test" => "helpers/test.md",
        ],
        "Contributing to Manopt.jl" => "contributing.md",
        "Extensions" => "extensions.md",
        "Notation" => "notation.md",
        "Changelog" => "changelog.md",
        "References" => "references.md",
    ],
    plugins = [bib, links],
)
deploydocs(; repo = "github.com/JuliaManifolds/Manopt.jl", push_preview = true)
#back to main env
Pkg.activate()
