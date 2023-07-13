#!/usr/bin/env julia
#
#
using Manopt
using Documenter#: DocMeta, HTML, MathJax3, deploydocs, makedocs
using DocumenterCitations

# (e) ...finally! make docs
bib = CitationBibliography(joinpath(@__DIR__, "src", "references.bib"); style=:authoryear)
print(bib)
makedocs(bib;
    modules=[Manopt],
    sitename="Manopt.jl",
    pages=[
        "Home" => "index.md",
    ],
)
deploydocs(; repo="github.com/JuliaManifolds/Manopt.jl", push_preview=true)
