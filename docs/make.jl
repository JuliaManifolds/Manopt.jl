using Documenter: DocMeta, HTML, MathJax3, deploydocs, makedocs
using Manopt, Manifolds, Literate, Pluto, PlutoStaticHTML, Pkg
# Load an unregistered package (for now) to update exports of Pluto notebooks

TutorialMenu = Array{Pair{String,String},1}()

#
# Generate Pluto Tutorial HTMLs

# First tutorial with AD
pluto_src_folder = joinpath(@__DIR__, "..", "pluto/")
pluto_output_folder = joinpath(@__DIR__, "src/", "pluto/")
pluto_relative_path = "pluto/"
mkpath(pluto_output_folder)
#
#
# Please do not use the same name as for a(n old) literate Tutorial
pluto_files = [
    "Optimize!",
    "AutomaticDifferentiation",
    "HowToRecord",
    "GeodesicRegression",
    "Bezier",
    "SecondOrderDifference",
    "StochasticGradientDescent",
    "Benchmark",
    "JacobiFields",
]
pluto_titles = [
    "Get started: Optimize!",
    "Use AD in Manopt",
    "How to record values",
    "Do Geodesic regression",
    "Use Bezier Curves",
    "Compute a second order difference",
    "Do stochastic gradient descent",
    "speed up! using `gradF!`",
    "Illustrate Jacobi Fields",
]
# build menu and write files myself - tp set edit url correctly.
for (title, file) in zip(pluto_titles, pluto_files)
    global TutorialMenu
    rendered = build_notebooks( #though not really parallel here
        BuildOptions(
            pluto_src_folder;
            output_format=documenter_output,
            write_files=false,
            use_distributed=false,
        ),
        ["$(file).jl"],
    )
    write(
        pluto_output_folder * file * ".md",
        """
        ```@meta
        EditURL = "$(pluto_src_folder)$(file).jl"
        ```

        $(rendered[1])
        """,
    )
    push!(TutorialMenu, title => joinpath(pluto_relative_path, file * ".md"))
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

makedocs(;
    format=HTML(; mathengine=MathJax3(), prettyurls=get(ENV, "CI", nothing) == "true"),
    modules=[Manopt],
    sitename="Manopt.jl",
    pages=[
        "Home" => "index.md",
        "About" => "about.md",
        "How to..." => TutorialMenu,
        "Plans" => [
            "Specify a Solver" => "plans/index.md",
            "Problem" => "plans/problem.md",
            "Options" => "plans/options.md",
            "Stepsize" => "plans/stepsize.md",
            "Stopping Criteria" => "plans/stopping_criteria.md",
        ],
        "Solvers" => [
            "Introduction" => "solvers/index.md",
            "Alternating Gradient Descent" => "solvers/alternating_gradient_descent.md",
            "Chambolle-Pock" => "solvers/ChambollePock.md",
            "Conjugate gradient descent" => "solvers/conjugate_gradient_descent.md",
            "Cyclic Proximal Point" => "solvers/cyclic_proximal_point.md",
            "Douglas–Rachford" => "solvers/DouglasRachford.md",
            "Gradient Descent" => "solvers/gradient_descent.md",
            "Nelder–Mead" => "solvers/NelderMead.md",
            "Particle Swarm Optimization" => "solvers/particle_swarm.md",
            "Quasi-Newton" => "solvers/quasi_Newton.md",
            "Stochastic Gradient Descent" => "solvers/stochastic_gradient_descent.md",
            "Subgradient method" => "solvers/subgradient.md",
            "Steihaug-Toint TCG Method" => "solvers/truncated_conjugate_gradient_descent.md",
            "Trust-Regions Solver" => "solvers/trust_regions.md",
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
