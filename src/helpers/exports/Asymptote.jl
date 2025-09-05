@doc """
    asymptote_export_S2_signals(filename; points, curves, tangent_vectors, colors, kwargs...)

Export given `points`, `curves`, and `tangent_vectors` on the sphere ``𝕊^2``
to Asymptote.

# Input
* `filename`          a file to store the Asymptote code in.

# Keywaord arguments for the data

* `colors=Dict{Symbol,Array{RGBA{Float64},1}}()`: dictionary of color arrays,
  indexed by symbols `:points`, `:curves` and `:tvector`, where each entry has to provide
  as least as many colors as the length of the corresponding sets.
* `curves=Array{Array{Float64,1},1}(undef, 0)`: an `Array` of `Arrays` of points
  on the sphere, where each inner array is interpreted as a curve
  and is accompanied by an entry within `colors`.
* `points=Array{Array{Float64,1},1}(undef, 0)`: an `Array` of `Arrays` of points
  on the sphere where each inner array is interpreted as a set of points and is accompanied
  by an entry within `colors`.
* `tangent_vectors=Array{Array{Tuple{Float64,Float64},1},1}(undef, 0)`:
  an `Array` of `Arrays` of tuples, where the first is a points, the second a tangent vector
  and each set of vectors is accompanied by an entry from within `colors`.

# Keyword arguments for asymptote

* `arrow_head_size=6.0`:
  size of the arrowheads of the tangent vectors
* `arrow_head_sizes`  overrides the previous value to specify a value per `tVector`` set.
* `camera_position=(1., 1., 0.)`:
  position of the camera in the Asymptote scene
* `line_width=1.0`:
  size of the lines used to draw the curves.
* `line_widths`       overrides the previous value to specify a value per curve and `tVector`` set.
* `dot_size=1.0`:
  size of the dots used to draw the points.
* `dot_sizes`         overrides the previous value to specify a value per point set.
* `size=nothing`:
  a tuple for the image size, otherwise a relative size `4cm` is used.
* `sphere_color=RGBA{Float64}(0.85, 0.85, 0.85, 0.6)`:
  color of the sphere the data is drawn on
* `sphere_line_color=RGBA{Float64}(0.75, 0.75, 0.75, 0.6)`:
  color of the lines on the sphere
* `sphere_line_width=0.5`:
  line width of the lines on the sphere
* `target=(0.,0.,0.)`:
  position the camera points at
"""
function asymptote_export_S2_signals(
        filename::String;
        points::Array{Array{T, 1}, 1} where {T} = Array{Array{Float64, 1}, 1}(undef, 0),
        curves::Array{Array{T, 1}, 1} where {T} = Array{Array{Float64, 1}, 1}(undef, 0),
        tangent_vectors::Array{Array{Tuple{T, T}, 1}, 1} where {T} = Array{
            Array{Tuple{Float64, Float64}, 1}, 1,
        }(
            undef, 0
        ),
        colors::Dict{Symbol, Array{RGBA{Float64}, 1}} = Dict{Symbol, Array{RGBA{Float64}, 1}}(),
        arrow_head_size::Float64 = 6.0,
        arrow_head_sizes::Array{Float64, 1} = fill(arrow_head_size, length(tangent_vectors)),
        camera_position::Tuple{Float64, Float64, Float64} = (1.0, 1.0, 0.0),
        line_width::Float64 = 1.0,
        line_widths::Array{Float64, 1} = fill(
            line_width, length(curves) + length(tangent_vectors)
        ),
        dot_size::Float64 = 1.0,
        dot_sizes::Array{Float64, 1} = fill(dot_size, length(points)),
        size::Union{Nothing, Tuple{Int, Int}} = nothing,
        sphere_color::RGBA{Float64} = RGBA{Float64}(0.85, 0.85, 0.85, 0.6),
        sphere_line_color::RGBA{Float64} = RGBA{Float64}(0.75, 0.75, 0.75, 0.6),
        sphere_line_width::Float64 = 0.5,
        target::Tuple{Float64, Float64, Float64} = (0.0, 0.0, 0.0),
    )
    io = open(filename, "w")
    return try
        #
        # Header
        # ---
        write(
            io,
            string(
                "import settings;\nimport three;\nimport solids;",
                isnothing(size) ? "unitsize(4cm);" : "size$(size);",
                "\n\n",
                "currentprojection=perspective( ",
                "camera = $(camera_position), ",
                "target = $(target) );\n",
                "currentlight=nolight;\n\n",
                "revolution S=sphere(O,0.995);\n",
                "pen SpherePen = rgb($(red(sphere_color)),",
                "$(green(sphere_color)),$(blue(sphere_color)))",
                "+opacity($(alpha(sphere_color)));\n",
                "pen SphereLinePen = rgb($(red(sphere_line_color)),",
                "$(green(sphere_line_color)),$(blue(sphere_line_color)))",
                "+opacity($(alpha(sphere_line_color)))+linewidth($(sphere_line_width)pt);\n",
                "draw(surface(S), surfacepen=SpherePen, meshpen=SphereLinePen);\n",
            ),
        )
        write(io, "\n/*\n  Colors\n*/\n")
        j = 0
        for (key, value) in colors # colors for all keys
            penPrefix = "$(j)"
            sets = 0
            if key == :points
                penPrefix = "point"
                sets = length(points)
            elseif key == :curves
                penPrefix = "curve"
                sets = length(curves)
            elseif key == :tvectors
                penPrefix = "tVector"
                sets = length(tangent_vectors)
            end
            if length(value) < sets
                throw(
                    ErrorException(
                        "Not enough colors ($(length(value))) provided for $(sets) sets in $(key).",
                    ),
                )
            end
            i = 0
            # export all colors
            for c in value
                i = i + 1
                if i > sets
                    # avoid access errors in `line_width` or `dot_sizes` if more colors then sets are given
                    break
                end
                write(
                    io,
                    string(
                        "pen $(penPrefix)Style$(i) = ",
                        "rgb($(red(c)),$(green(c)),$(blue(c)))",
                        (key == :curves) ? "+linewidth($(line_widths[i])pt)" : "",
                        if (key == :tvectors)
                            "+linewidth($(line_widths[length(curves) + i])pt)"
                        else
                            ""
                        end,
                        (key == :points) ? "+linewidth($(dot_sizes[i])pt)" : "",
                        "+opacity($(alpha(c)));\n",
                    ),
                )
            end
        end
        if length(points) > 0
            write(io, "\n/*\n  Exported Points\n*/\n")
        end
        i = 0
        for pSet in points
            i = i + 1
            for point in pSet
                write(
                    io,
                    string(
                        "dot( (",
                        string([string(v, ",") for v in point]...)[1:(end - 1)],
                        "), pointStyle$(i));\n",
                    ),
                )
            end
        end
        i = 0
        if length(curves) > 0
            write(io, "\n/*\n  Exported Curves\n*/\n")
        end
        for curve in curves
            i = i + 1
            write(io, "path3 p$(i) = ")
            j = 0
            for point in curve
                j = j + 1
                pString = "(" * string(["$v," for v in point]...)[1:(end - 1)] * ")"
                write(io, j > 1 ? " .. $(pString)" : pString)
            end
            write(io, string(";\n draw(p$(i), curveStyle$(i));\n"))
        end
        i = 0
        if length(tangent_vectors) > 0
            write(io, "\n/*\n  Exported tangent vectors\n*/\n")
        end
        for tVecs in tangent_vectors
            i = i + 1
            j = 0
            for vector in tVecs
                j = j + 1
                base = vector[1]
                endPoints = base + vector[2]
                write(
                    io,
                    string(
                        "draw( (",
                        string([string(v, ",") for v in base]...)[1:(end - 1)],
                        ")--(",
                        string([string(v, ",") for v in endPoints]...)[1:(end - 1)],
                        "), tVectorStyle$(i),Arrow3($(arrow_head_sizes[i])));\n",
                    ),
                )
            end
        end
    finally
        close(io)
    end
end
@doc """
    asymptote_export_S2_data(filename)

Export given `data` as an array of points on the 2-sphere, which might be one-, two-
or three-dimensional data with points on the [Sphere](https://juliamanifolds.github.io/Manifolds.jl/stable/manifolds/sphere.html) ``𝕊^2``.

# Input

* `filename`                a file to store the Asymptote code in.

# Optional arguments for the data

* `data`                    a point representing the 1D,2D, or 3D array of points
* `elevation_color_scheme`  A `ColorScheme` for elevation
* `scale_axes=(1/3,1/3,1/3)`:
  move spheres closer to each other by a factor
  per direction

# Optional arguments for asymptote

* `arrow_head_size=1.8`:
  size of the arrowheads of the vectors (in mm)
* `camera_position`  position of the camera scene (default: atop the center of the data in the xy-plane)
* `target`           position the camera points at (default: center of xy-plane within data).
"""
function asymptote_export_S2_data(
        filename::String;
        data = fill([0.0, 0.0, 1.0], 0, 0),
        arrow_head_size::Float64 = 1.8,
        scale_axes = (1 / 3.0, 1 / 3.0, 1 / 3.0),
        camera_position::Tuple{Float64, Float64, Float64} = scale_axes .* (
            (size(data, 1) - 1) / 2, (size(data, 2) - 1) / 2, max(size(data, 3), 0) + 10,
        ),
        target::Tuple{Float64, Float64, Float64} = scale_axes .* (
            (size(data, 1) - 1) / 2, (size(data, 2) - 1) / 2, 0.0,
        ),
        elevation_color_scheme = ColorSchemes.viridis,
    )
    io = open(filename, "w")
    return try
        write(
            io,
            string(
                "import settings;\nimport three;\n",
                "size(7cm);\n",
                "DefaultHead.size=new real(pen p=currentpen) {return $(arrow_head_size)mm;};\n",
                "currentprojection=perspective( ",
                "camera = $(camera_position), up=Y,",
                "target = $(target) );\n\n",
            ),
        )
        dims = [size(data, i) for i in [1, 2, 3]]
        for x in 1:dims[1]
            for y in 1:dims[2]
                for z in 1:dims[3]
                    v = Tuple(data[x, y, z]) #extract value
                    el = asin(min(1, max(-1, v[3]))) # since 3 is between -1 and 1 this yields a value between 0 and pi
                    # map elevation to color map
                    c = get(elevation_color_scheme, el + π / 2, (0.0, Float64(π)))
                    # write arrow in this color map
                    # transpose image to comply with image addresses (first index column downwards, second rows)
                    write(
                        io,
                        string(
                            "draw( $(scale_axes .* (x - 1, y - 1, z - 1))",
                            "--$(scale_axes .* (x - 1, y - 1, z - 1) .+ v),",
                            " rgb($(red(c)),$(green(c)),$(blue(c))), Arrow3);\n",
                        ),
                    )
                end
            end
        end
    finally
        close(io)
    end
end
@doc """
    asymptote_export_SPD(filename)

export given `data` as a point on a `Power(SymmetricPOsitiveDefinnite(3))}` manifold of
one-, two- or three-dimensional data with points on the manifold of symmetric positive
definite matrices.

# Input
* `filename`        a file to store the Asymptote code in.

# Optional arguments for the data

* `data`            a point representing the 1D, 2D, or 3D array of SPD matrices
* `color_scheme`    a `ColorScheme` for Geometric Anisotropy Index
* `scale_axes=(1/3,1/3,1/3)`:
  move symmetric positive definite matrices
  closer to each other by a factor per direction compared to the distance
  estimated by the maximal eigenvalue of all involved SPD points

# Optional arguments for asymptote

* `camera_position`  position of the camera scene (default: atop the center of the data in the xy-plane)
* `target`           position the camera points at (default: center of xy-plane within data).

Both values `camera_position` and `target` are scaled by `scaledAxes*EW`, where
`EW` is the maximal eigenvalue in the `data`.
"""
function asymptote_export_SPD(
        filename::String;
        data = fill(Matrix{Float64}(I, 3, 3), 0, 0),
        scale_axes = (1 / 3.0, 1 / 3.0, 1 / 3.0) .*
            (length(data) > 0 ? maximum(maximum(eigvals.(data))) : 1),
        camera_position::Tuple{Float64, Float64, Float64} = (
            (size(data, 1) - 1) / 2, (size(data, 2) - 1) / 2, max(size(data, 3), 0.0) + 10.0,
        ),
        target::Tuple{Float64, Float64, Float64} = (
            (size(data, 1) - 1) / 2, (size(data, 2) - 1) / 2, 0.0,
        ),
        color_scheme = ColorSchemes.viridis,
    )
    io = open(filename, "w")
    return try
        write(
            io,
            string(
                "import settings;\nimport three;\n",
                "surface ellipsoid(triple v1,triple v2,triple v3,real l1,real l2, real l3, triple pos=O) {\n",
                "  transform3 T = identity(4);\n",
                "  T[0][0] = l1*v1.x;\n  T[1][0] = l1*v1.y;\n  T[2][0] = l1*v1.z;\n",
                "  T[0][1] = l2*v2.x;\n  T[1][1] = l2*v2.y;\n  T[2][1] = l2*v2.z;\n",
                "  T[0][2] = l3*v3.x;\n  T[1][2] = l3*v3.y;\n  T[2][2] = l3*v3.z;\n",
                "  T[0][3] = pos.x;\n  T[1][3] = pos.y;\n  T[2][3] = pos.z;\n",
                "  return T*unitsphere;\n}\n\n",
                "size(200);\n\n",
                "real gDx=$(scale_axes[1]);\n",
                "real gDy=$(scale_axes[2]);\n",
                "real gDz=$(scale_axes[3]);\n\n",
                "currentprojection=perspective(up=Y, ",
                "camera = (gDx*$(camera_position[1]),gDy*$(camera_position[2]),gDz*$(camera_position[3])), ",
                "target = (gDx*$(target[1]),gDy*$(target[2]),gDz*$(target[3])) );\n",
                "currentlight=Viewport;\n\n",
            ),
        )
        dims = [size(data, 1) size(data, 2) size(data, 3)]
        for x in 1:dims[1]
            for y in 1:dims[2]
                for z in 1:dims[3]
                    A = data[x, y, z] #extract matrix
                    F = eigen(A)
                    if maximum(abs.(A)) > 0.0 # a nonzero matrix (exclude several pixel
                        # Following Moakher & Batchelor: Geometric Anisotropic Index:
                        λ = F.values
                        V = F.vectors
                        Lλ = log.(λ)
                        GAI = sqrt(
                            2 / 3 * sum(Lλ .^ 2) -
                                2 / 3 * sum(sum(tril(Lλ * Lλ', -1); dims = 1); dims = 2)[1],
                        )
                        c = get(color_scheme, GAI / (1 + GAI), (0, 1))
                        write(
                            io,
                            string(
                                "  draw(  ellipsoid( ($(V[1, 1]),$(V[2, 1]),$(V[3, 1])),",
                                " ($(V[1, 2]),$(V[2, 2]),$(V[3, 2])), ($(V[1, 3]),$(V[2, 3]),$(V[3, 3])),",
                                " $(λ[1]), $(λ[2]), $(λ[3]), ",
                                " (gDx*$(x - 1), gDy*$(y - 1), gDz*$(z - 1))),",
                                " rgb($(red(c)),$(green(c)),$(blue(c)))  );\n",
                            ),
                        )
                    end
                end
            end
        end
    finally
        close(io)
    end
end

"""
    render_asymptote(filename; render=4, format="png", ...)
render an exported asymptote file specified in the `filename`, which can also
be given as a relative or full path

# Input

* `filename`    filename of the exported `asy` and rendered image

# Keyword arguments

the default values are given in brackets

* `render=4`:
  render level of asymptote passed to its `-render` option.
   This can be removed from the command by setting it to `nothing`.
* `format="png"`:
  final rendered format passed to the `-f` option
* `export_file`: (the filename with format as ending) specify the export filename
"""
function render_asymptote(
        filename;
        render::Union{Int, Nothing} = 4,
        format = "png",
        export_folder = string(filename[1:([findlast(".", filename)...][1])], format),
    )
    if isnothing(render)
        renderCmd = `asy -f $(format) -globalwrite  -o "$(relpath(export_folder))" $(filename)`
    else
        renderCmd = `asy -render $(render) -f $(format) -globalwrite  -o "$(relpath(export_folder))" $(filename)`
    end
    return run(renderCmd)
end
