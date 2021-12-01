using Manopt, Manifolds, Documenter, Literate, Pluto, PlutoStaticHTML
# Load an unregistered package (for now) to update exports of Pluto notebooks

# generate examples using Literate
tutorialsInputPath = joinpath(@__DIR__, "..", "src/tutorials")
tutorialsRelativePath = "tutorials/"
tutorialsOutputPath = joinpath(@__DIR__, "src/" * tutorialsRelativePath)
tutorials = [
    "MeanAndMedian",
    "Benchmark",
    "GeodesicRegression",
    "HowToRecord",
    "StochasticGradientDescent",
    "BezierCurves",
    "GradientOfSecondOrderDifference",
    "JacobiFields",
]
menuEntries = [
    "get Started: Optimize!",
    "speed up! using `gradF!`",
    "Do Geodesic regression",
    "Record values",
    "do stochastic gradient descent",
    "work with Bézier curves",
    "see the gradient of ``d_2``",
    "use Jacobi Fields",
]
TutorialMenu = Array{Pair{String,String},1}()
for (i, tutorial) in enumerate(tutorials)
    global TutorialMenu
    sourceFile = joinpath(tutorialsInputPath, tutorial * ".jl")
    targetFile = joinpath(tutorialsOutputPath, tutorial * "md")
    Literate.markdown(sourceFile, tutorialsOutputPath; name=tutorial, credit=false)
    push!(TutorialMenu, menuEntries[i] => joinpath(tutorialsRelativePath, tutorial * ".md"))
end

#
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
pluto_files = ["AutomaticDifferentiation"]
pluto_heights = [370] # for now, lazyness, in rem
pluto_titles = ["AD in Manopt"]
for (i, f) in enumerate(pluto_files)
    global TutorialMenu
    @info "Building Pluto Notebook $f.jl"
    pluto_file = pluto_src_folder * f * ".jl"
    html = notebook2html(pluto_file)
    write(
        pluto_output_folder * f * ".md",
        """
        ```@meta
        EditURL = "$(pluto_src_folder)$(f).jl"
        ```

        ```@raw html
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/fonsp/Pluto.jl@0.16.4/frontend/treeview.css" type="text/css" />
        <style>
        div.markdown {
            padding-top: 1rem;
            padding-bottom: 2rem;
        }
        </style>
        $html
        <script>
  </script>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.15.1/dist/katex.min.css" integrity="sha384-R4558gYOUz8mP9YWpZJjofhk+zx0AS11p36HnD2ZKj/6JR5z27gSSULCNHIRReVs" crossorigin="anonymous">

    <!-- The loading of KaTeX is deferred to speed up page rendering -->
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.15.1/dist/katex.min.js" integrity="sha384-z1fJDqw8ZApjGO3/unPWUPsIymfsJmyrDVWC8Tv/a1HeOtGmkwNd/7xUS0Xcnvsx" crossorigin="anonymous"></script>

    <!-- To automatically render math in text elements, include the auto-render extension: -->
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.15.1/dist/contrib/auto-render.min.js" integrity="sha384-+XBljXPPiv+OzfbB3cVmLHf4hdUFHlWNZN5spNQ7rmHTXpd7WvJum6fIACpNNfIR" crossorigin="anonymous"
        onload="renderMathInElement(document.body);"></script>
        
  <script>
  const options = {
    delimiters: [
      {left: "\$\$", right: "\$\$", display: true},
      {left: "\$", right: "\$", display: false},
      {left: "\\begin{equation}", right: "\\end{equation}", display: true},
      {left: "\\begin{align}", right: "\\end{align}", display: true},
      {left: "\\begin{alignat}", right: "\\end{alignat}", display: true},
      {left: "\\begin{gather}", right: "\\end{gather}", display: true},
      {left: "\\(", right: "\\)", display: false},
      {left: "\\[", right: "\\]", display: true}
    ]
  };
  renderMathInElement(document.body, options);
  </script>
        </script>
        ```
        """,
    )
    push!(TutorialMenu, pluto_titles[i] => joinpath(pluto_relative_path, f * ".md"))
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
    format=Documenter.HTML(; prettyurls=false),
    modules=[Manopt],
    sitename="Manopt.jl",
    pages=[
        "Home" => "index.md",
        "About" => "about.md",
        "How to..." => TutorialMenu,
        "Plans" => "plans/index.md",
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
