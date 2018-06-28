using Manopt, Documenter

makedocs()

deploydocs(
    target = "site",
    repo   = "github.com/kellertuer/Manopt.jl",
    branch = "gh-pages",
    latest = "master",
    osname = "osx",
    julia  = "0.6"
)
