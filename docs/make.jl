using Manopt, Documenter

makedocs(
    # for development, we disable prettyurls
    format = Documenter.HTML(prettyurls = false),
    modules=[Manopt],
    sitename = "Manopt.jl",
    pages = [
        "Home" => "index.md",
        "Manifolds" => [
            "Introduction" => "manifolds/index.md",
            "Combinations of Manifolds" => "manifolds/combined.md",
            "The Circle \$\\mathbb S^1\$" => "manifolds/circle.md",
            "The Euclidean Space \$\\mathbb R^n\$" => "manifolds/euclidean.md",
            "The Hyperbolic Space \$\\mathbb H^n\$" => "manifolds/hyperbolic.md",
            "The Sphere \$\\mathbb S^n\$" => "manifolds/sphere.md",
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
        "Function Index" => "list.md"
    ]
)
#deploydocs(
#     target = "site",
#     repo   = "github.com/kellertuer/Manopt.jl",
#     branch = "gh-pages",
#     latest = "master",
#     osname = "osx",
#     julia  = "1.0",
#     deps = Deps.pip("pygments", "mkdocs", "python-markdown-math")
#)
