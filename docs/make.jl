using Documenter: DocMeta, HTML, MathJax3, deploydocs, makedocs
using Manopt, Manifolds, Literate, Pluto, PlutoStaticHTML, Pkg
# Load an unregistered package (for now) to update exports of Pluto notebooks

struct Notebook
    name::String
    title::String
end

function build(dir::String, notebooks::Vector{Notebook})
    bopts = BuildOptions(
        dir;
        output_format=documenter_output,
        write_files=true,
        use_distributed=true
    )
    files = [nb.name * ".jl" for nb in notebooks]
    build_notebooks(bopts, files)
    return nothing
end

@info " \n      Rendering Tutorials\n "

tutorial_src_folder = joinpath(@__DIR__, "..", "tutorials/")
tutorial_output_folder = joinpath(@__DIR__, "src/", "tutorials/")
tutorial_relative_path = "tutorials/"
mkpath(tutorial_output_folder)

tutorials = Notebook[
    Notebook("Optimize!", "Get Started: Optimize!"),
    Notebook("AutomaticDifferentiation", "Use AD in Manopt"),
    Notebook("HowToRecord", "Record Values"),
    Notebook("ConstrainedOptimization", "Do constrained Optimization"),
    Notebook("GeodesicRegression", "Do Geodesic Regression"),
    Notebook("Bezier", "Use Bézier Curves"),
    Notebook("SecondOrderDifference", "Compute a Second Order Difference"),
    Notebook("StochasticGradientDescent", "Do Stochastic Gradient Descent"),
    Notebook("Benchmark", "Speed up! Using `gradF!`"),
    Notebook("JacobiFields", "Illustrate Jacobi Fields"),
]

build(tutorial_src_folder, tutorials)

tutorial_menu = map(tutorials) do nb::Notebook
    nb.title => joinpath(tutorial_relative_path, nb.name * ".md")
end

@info " \n      Rendering Examples\n "

examples_src_folder = joinpath(@__DIR__, "..", "examples/")
examples_output_folder = joinpath(@__DIR__, "src/", "examples/")
examples_relative_path = "examples/"
mkpath(examples_output_folder)

examples = [
    Notebook("robustPCA", "Robust PCA"),
    Notebook("smallestEigenvalue", "Rayleigh quotient"),
    Notebook("FrankWolfeSPDMean", "Frank Wolfe for Riemannian Center of Mass")
]

build(examples_src_folder, examples)

example_menu = map(examples) do nb::Notebook
    nb.title => joinpath(examples_relative_path, nb.name * ".md")
end

generated_path = joinpath(@__DIR__, "src")
base_url = "https://github.com/JuliaManifolds/Manopt.jl/blob/master/"
mkpath(generated_path)
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

@info " \n      Rendering Documentation\n "

makedocs(;
    format=HTML(; mathengine=MathJax3(), prettyurls=get(ENV, "CI", nothing) == "true"),
    modules=[Manopt],
    sitename="Manopt.jl",
    pages=[
        "Home" => "index.md",
        "About" => "about.md",
        "How to..." => tutorial_menu,
        "Solvers" => [
            "Introduction" => "solvers/index.md",
            "Alternating Gradient Descent" => "solvers/alternating_gradient_descent.md",
            "Augmented Lagrangian Method" => "solvers/augmented_Lagrangian_method.md",
            "Chambolle-Pock" => "solvers/ChambollePock.md",
            "Conjugate gradient descent" => "solvers/conjugate_gradient_descent.md",
            "Cyclic Proximal Point" => "solvers/cyclic_proximal_point.md",
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
        "Examples" => example_menu,
        "Plans" => [
            "Specify a Solver" => "plans/index.md",
            "Problem" => "plans/problem.md",
            "Options" => "plans/options.md",
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
            "Adjoint Differentials" => "functions/adjointdifferentials.md",
            "Gradients" => "functions/gradients.md",
            "Jacobi Fields" => "functions/Jacobi_fields.md",
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
        "Notation" => "notation.md",
        "Function Index" => "list.md",
    ],
)
deploydocs(; repo="github.com/JuliaManifolds/Manopt.jl", push_preview=true)
