using Manopt, Documenter, Literate

# generate examples using Literate
tutorialsInputPath = joinpath(@__DIR__, "..", "src/tutorials")
tutorialsRelativePath = "tutorials/"
tutorialsOutputPath = joinpath(@__DIR__, "src/" * tutorialsRelativePath)
tutorials = ["MeanAndMedian", "BezierCurves","GradientOfSecondOrderDifference", "JacobiFields"]
menuEntries = ["get Started: Optimize!", "work with Bézier curves","see the gradient of \$d_2\$", "use Jacobi Fields"]
TutorialMenu = Array{Pair{String,String},1}()
for (i, tutorial) in enumerate(tutorials)
    global TutorialMenu
    sourceFile = joinpath(tutorialsInputPath, tutorial * ".jl")
    targetFile = joinpath(tutorialsOutputPath, tutorial * "md")
    Literate.markdown(
        sourceFile,
        tutorialsOutputPath;
        name = tutorial,
        # codefence = "```julia" => "```",
        credit = false,
    )
    push!(TutorialMenu, menuEntries[i] => joinpath(tutorialsRelativePath, tutorial * ".md"))
end
makedocs(
    format = Documenter.HTML(prettyurls = false),
    modules = [Manopt],
    sitename = "Manopt.jl",
    pages = [
        "Home" => "index.md",
        "About" => "about.md",
        "How to..." => TutorialMenu,
        "Plans" => "plans/index.md",
        "Solvers" => [
            "Introduction" => "solvers/index.md",
            "Conjugate gradient descent" => "solvers/conjugate_gradient_descent.md",
            "Cyclic Proximal Point" => "solvers/cyclic_proximal_point.md",
            "Douglas–Rachford" => "solvers/DouglasRachford.md",
            "Gradient Descent" => "solvers/gradientDescent.md",
            "Nelder–Mead" => "solvers/NelderMead.md",
            "Particle Swarm Optimization" => "solvers/particle_swarm.md",
            "Subgradient method" => "solvers/subgradient.md",
            "Steihaug-Toint TCG Method" =>
                "solvers/truncated_conjugate_gradient_descent.md",
            "Riemannian Trust-Regions Solver" => "solvers/trust_regions.md",
        ],
        "Functions" => [
            "Introduction" => "functions/index.md",
            "Bézier curves" => "functions/bezier.md",
            "Cost functions" => "functions/costFunctions.md",
            "Differentials" => "functions/differentials.md",
            "Adjoint Differentials" => "functions/adjointDifferentials.md",
            "Gradients" => "functions/gradients.md",
            "JacobiFields" => "functions/jacobiFields.md",
            "Proximal Maps" => "functions/proximalMaps.md",
            "Specific manifold functions" => "functions/manifold.md",
        ],
        "Helpers" => [
            "Data" => "helpers/data.md",
            "Error Measures" => "helpers/errorMeasures.md",
            "Exports" => "helpers/exports.md",
        ],
        "Function Index" => "list.md",
    ],
)
deploydocs(repo = "github.com/JuliaManifolds/Manopt.jl", push_preview = true)
