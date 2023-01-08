using Documenter: DocMeta, HTML, MathJax3, deploydocs, makedocs
using Manopt, Manifolds, Literate, Pluto, PlutoStaticHTML, Pkg
# Load an unregistered package (for now) to update exports of Pluto notebooks

tutorial_menu = Array{Pair{String,String},1}()

#
# Generate Pluto Tutorial HTMLs
tutorial_src_folder = joinpath(@__DIR__, "..", "tutorials/")
tutorial_output_folder = joinpath(@__DIR__, "src/", "tutorials/")
tutorial_relative_path = "tutorials/"
mkpath(tutorial_output_folder)
#
# Tutorials
@info " \n      Rendering Tutorials"
tutorials = [
    Dict(:file => "Optimize!", :title => "Get Started: Optimize!"),
    Dict(:file => "AutomaticDifferentiation", :title => "Use AD in Manopt"),
    Dict(:file => "HowToRecord", :title => "Record Values"),
    Dict(:file => "ConstrainedOptimization", :title => "Do constrained Optimization"),
    Dict(:file => "GeodesicRegression", :title => "Do Geodesic Regression"),
    Dict(:file => "Bezier", :title => "Use Bézier Curves"),
    Dict(:file => "SecondOrderDifference", :title => "Compute a Second Order Difference"),
    Dict(:file => "StochasticGradientDescent", :title => "Do Stochastic Gradient Descent"),
    Dict(:file => "Benchmark", :title => "Speed up! Using `gradF!`"),
    Dict(:file => "JacobiFields", :title => "Illustrate Jacobi Fields"),
]
# build menu and write files myself - tp set edit url correctly.
for t in tutorials
    global tutorial_menu
    rendered = build_notebooks( #though not really parallel here
        BuildOptions(
            tutorial_src_folder;
            output_format=documenter_output,
            write_files=false,
            use_distributed=true,
            previous_dir=tutorial_output_folder,
        ),
        ["$(t[:file]).jl"],
    )
    write(
        tutorial_output_folder * t[:file] * ".md",
        """
        ```@meta
        EditURL = "$(tutorial_src_folder)$(t[:file]).jl"
        ```
        $(rendered["$(t[:file]).jl"][1])
        """,
    )
    push!(tutorial_menu, t[:title] => joinpath(tutorial_relative_path, t[:file] * ".md"))
end

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

@info " \n      Rendering Documentation"
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
            "Adjoint Differentials" => "functions/adjointdifferentials.md",
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
        "Notation" => "notation.md",
        "Function Index" => "list.md",
    ],
)
deploydocs(; repo="github.com/JuliaManifolds/Manopt.jl", push_preview=true)
