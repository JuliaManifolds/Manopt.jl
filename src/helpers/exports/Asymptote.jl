using ColorTypes, Colors
export asyExportS2

function asyExportS2(filename::String;
    points::Array{SnPoint,2} = Array{SnPoint,2}(undef,0,0),
    curves::Array{SnPoint,2} = Array{SnPoint,2}(undef,0,0),
    tVectors::Array{TVectorE{SnTVector,SnPoint},1} = Array{TVectorE{SnTVector,SnPoint},1}(undef,0),
    colors::Dict{Symbol, Array{T,1} where T}  = Dict{Symbol,Array{RGBA{Float},1}}(),
    arrowHeadSize::Float64 = 6.,
    cameraPosition::Tuple{Float64,Float64,Float64} = (1., 1., 0.),
    dotSize::Float64 = 1.0,
    dotSizes::Union{Array{Float64,1},Missing} = missing,
    target::Tuple{Float64,Float64,Float64} = (0.,0.,0.),
    )
    if ismissing(dotSizes)
        dotSizes = fill(dotSize,size(points))
    end
    io = open(filename,"w")
    try
        #
        # Header
        # ---
        write(io,string("import settings;\nimport three;\nimport solids;",
                    "unitsize(4cm);\n\n",
                    "currentprojection=perspective( ",
                    "camera = $(cameraPosition), ",
                    "target = $(target) );\n",
                    "currentlight=nolight;\n\n",
                    "revolution S=sphere(O,1);\n",
                    "draw(surface(S), surfacepen=lightgrey+opacity(.6), ",
                    "meshpen=0.6*white+linewidth(.5pt));\n")
        );
        write(io,"\n/*\n  Colors\n*/\n")
        j=0
        for (key,value) in colors
            penPrefix = "$(j)"
            penPrefix = "$(j)"
            if key==:points
                penPrefix="point"
            elseif key==:curves
                penPrefix="curve"
            elseif key==:tvectors
                penPrefix="tVector"
            end
            i=0
            for c in value
                i=i+1;
                write(io,string("pen $(penPrefix)Style$(i) ",
                    "rgb($(red(c)),$(green(c)),$(blue(c)))",
                    "+lineWidth($(dotSize[i]))",
                    "+opacity($(alpha(c)));\n"));
            end
        end
        write(io,"\n/*\n  Exported Points\n*/\n")
        i=0
        for pSet in points
            i=i+1
            for point in pSet
                write(io,string("dot($(getValue(point)), pointStyle$(i));\n"));
            end
        end
        i=0
        for curve in curves
            i=i+1
            write(io,"path3 p$(i) = ");
            j=0
            for point in pSet
                j=j+1
                if j>1
                    write( io," .. $(getValue(point))" )
                else
                    write( io,"$(getValue(point))" )
                end
                write( io,string(";\n draw(p$(i), curveStyle$(i));\n") );
            end
        end
        i=0
        for tVecs in tVectors
            i=i+1
            j=0
            for vector in tVecs
                j=j+1
                write(io,string("draw($(getBase(vector))--$(getBase(vector)+getValue(vector)),",
                    "tVectorStyle$(j),Arrow3($(arrowHeadSize)));\n"));
            end
        end
    finally
        close(io)
    end
    # for i=1:m %linestyles
    #     lColor = vars.colors{i};
    #     line = sprintf(['\npen LineStyle',num2str(i),' = ',...
    #         'rgb(',num2str(lColor(1)),',',num2str(lColor(2)),',',num2str(lColor(3)),')+',...
    #         'linewidth(',num2str(dS(i)),')+opacity(',num2str(vars.OpacityVector(i)),');']);
    #     if ~isempty(vars.File)
    #         fprintf(fID,line);
    #     end
    #     lfileStr = [lfileStr,line]; %#ok<*AGROW>
    # end
    # % for the signals, 1,...,s
    # for i=1:s %for each signal
    #     lSig = vars.pts{i};
    #     l = size(lSig,2);
    #     for j=1:l %for all dots each
    #         line = sprintf(['dot((',num2str(lSig(1,j)),',',num2str(lSig(2,j)),',',num2str(lSig(3,j)),'),LineStyle',num2str(i),');\n']);
    #         if ~isempty(vars.File)
    #             fprintf(fID,line);
    #         end
    #         lfileStr = [lfileStr,line];
    #     end
    # end
    # % for curves
    # for i=1:t
    #     lCurve = vars.curves{i};
    #     l = size(lCurve,2);
    #     line = ['path3 p',num2str(i),' = '];
    #     for j=1:l %for all dots each
    #         if j > 1
    #             line = sprintf([line,' .. ']);
    #         end
    #         line = sprintf([line,'(',num2str(lCurve(1,j)),',',num2str(lCurve(2,j)),',',num2str(lCurve(3,j)),')']);
    #     end
    #     line = sprintf([line,';\ndraw(p',num2str(i),',LineStyle',num2str(s+i),');\n']);
    #     if ~isempty(vars.File)
    #         fprintf(fID,line);
    #     end
    #     lfileStr = [lfileStr,line];
    # end
    # % for tangent vectors
    # for i=1:u
    #     lxi = vars.xi{i};
    #     l = size(lxi,3);
    #     for j=1:l
    #         lineS = sprintf(['draw((',num2str(lxi(1,1,j)),',',num2str(lxi(2,1,j)),',',num2str(lxi(3,1,j))... %base vec
    #             ,')--(',num2str(lxi(1,1,j)+lxi(1,2,j)),',',num2str(lxi(2,1,j)+lxi(2,2,j)),',',num2str(lxi(3,1,j)+lxi(3,2,j))...
    #             ,'),LineStyle',num2str(s+t+i),',Arrow3(',num2str(vars.ArrowHead),'));\n']);
    #          if ~isempty(vars.File)
    #             fprintf(fID,lineS);
    #          end
    #         lfileStr = [lfileStr,lineS];
    #     end
    # end
end
