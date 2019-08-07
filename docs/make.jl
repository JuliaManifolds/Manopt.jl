using Manopt, Documenter, Literate

# generate examples using Literate
tutorialsInputPath = joinpath(@__DIR__, "..", "src/tutorials")
tutorialsRelativePath = "tutorials/"
tutorialsOutputPath = joinpath(@__DIR__,"src/"*tutorialsRelativePath)
tutorials  = [
    "MeanAndMedian",
    "GradientOfSecondOrderDifference",
    "JacobiFields",
    ]
menuEntries = [
    "Getting Started: Optimize!",
    "Gradient of \$d_2\$",
    "Jacobi Fields"]
TutorialMenu = Array{Pair{String,String},1}()
for (i,tutorial) in enumerate(tutorials)
    global TutorialMenu
    sourceFile = joinpath(tutorialsInputPath,tutorial*".jl")
    targetFile = joinpath(tutorialsOutputPath,tutorial*"md")
    Literate.markdown(sourceFile,tutorialsOutputPath; name=tutorial,
    # codefence = "```julia" => "```",
    credit=false)
    push!(TutorialMenu, menuEntries[i] => joinpath(tutorialsRelativePath,tutorial*".md") )
end
makedocs(
    # for development, we disable prettyurls
    # format = Documenter.HTML(prettyurls = false),
    modules = [Manopt],
    sitename = "Manopt.jl",
    pages = [
        "Home" => "index.md",
        "About" => "about.md",
        "Manifolds" => [
            "Introduction" => "manifolds/index.md",
            "Combinations of Manifolds" => "manifolds/combined.md",
            "The Circle \$\\mathbb S^1\$" => "manifolds/circle.md",
            "The Euclidean Space \$\\mathbb R^n\$" => "manifolds/euclidean.md",
            "The Grassmannian Manifold \$\\mathrm{Gr}(k,n)\$" => "manifolds/grassmannian.md",
            "The Hyperbolic Space \$\\mathbb H^n\$" => "manifolds/hyperbolic.md",
            "The Special Orthogonal Group \$\\mathrm{SO}(n)\$" => "manifolds/rotations.md",
            "The Sphere \$\\mathbb S^n\$" => "manifolds/sphere.md",
            "The Stiefel Manifold \$\\mathrm{St}(k,n)\$" => "manifolds/stiefel.md",
            "The Symmetric Matrices \$\\mathrm{Sym}(n)\$" => "manifolds/symmetric.md",
            "The Symmetric Positive Definite Matrices \$\\mathcal P(n)\$" => "manifolds/symmetricpositivedefinite.md",
        ],
        "Plans" => "plans/index.md",
        "Solvers" => [
            "Introduction" => "solvers/index.md",
            "Cyclic Proximal Point" => "solvers/cyclicProximalPoint.md",
            "Douglas–Rachford" => "solvers/DouglasRachford.md",
            "Gradient Descent" => "solvers/gradientDescent.md",
            "Subgradient Method" => "solvers/subGradientMethod.md",
         ],
        "Functions" => [
            "Introduction" => "functions/index.md",
            "cost functions" => "functions/costFunctions.md",
            "Differentials" => "functions/differentials.md",
            "Adjoint Differentials" => "functions/adjointDifferentials.md",
            "Gradients" => "functions/gradients.md",
            "JacobiFields" => "functions/jacobiFields.md",
            "Proximal Maps" => "functions/proximalMaps.md"
        ],
        "Helpers" => [
            "Data" => "helpers/data.md",
            "Error Measures" => "helpers/errorMeasures.md",
            "Exports" => "helpers/exports.md",
        ],
        "Tutorials" => TutorialMenu,
        "Function Index" => "list.md",
    ]
)
deploydocs(
    repo   = "github.com/kellertuer/Manopt.jl",
   # devbranch = "development"
)
