using Manopt, Documenter

makedocs(
    format=:html,
    modules=[Manopt],
    sitename = "Manopt.jl",
    pages = [
        "Home" => "index.md",
        "Manifolds" => [
            "Introduction" => "manifolds/index.md",
            "The Sphere \$\\mathbb S^n\$" => "manifolds/sphere.md",
        ],
        "Plans" => [
            "Introduction" => "plans/index.md"
        ],
        "Solvers" => [
            "Introduction" => "solvers/index.md"
        ]
    ]
)
#deploydocs(
#     target = "site",
#     repo   = "github.com/kellertuer/Manopt.jl",
#     branch = "gh-pages",
#     latest = "master",
#     osname = "osx",
#     julia  = "0.6",
#     deps = Deps.pip("pygments", "mkdocs", "python-markdown-math")
#)
