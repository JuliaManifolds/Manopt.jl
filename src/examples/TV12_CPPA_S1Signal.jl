# Denoising a phase-valued Signal
# ---
# This is example is adopted (different noise) and extended from Example 5.1 in
# > Bergmann, R., Laus, F., Steidl, G. and Weinmann, A.,
# > Second order differences of cyclic data and applications in variational denoising.
# > SIAM Journal on Imaging Sciences, 7(4), pp. 2916–53, 2014
# > doi 10.1137/140969993, arXiv 1405.5349
#
# or the tutorial “Denoise a Phase-valued Signal” for the Jupyter notebook.

using Manopt
using Plots
using LaTeXStrings
using DataFrames
using CSV
resultsFolder = string(@__DIR__,"/results_TV12_CPPA_S1Signal/")
fileType = ".pdf"

# Create Data
x = 0:0.002:1
y = S1Signal.(x,true);
yR = S1Signal.(x,false);

# Put data in manifold-types
M = Circle()
yS = S1Point.(y)
ySn = addNoise.(Ref(M),yS,0.2);
# noisy data array
yn = getValue.(ySn);
# Power manifold
yPow = PowPoint(yS)
MPow = Power(M,size(yn))
ynPow = PowPoint(ySn);

plot(x,yR,label="real",
    linetype=:scatter, marker=(2.5,stroke(0)),
    yticks=([-π,-π/2,0,π/2,π],[L"-\pi",L"-\frac{\pi}{2}",0,L"\frac{\pi}{2}",L"\pi"]),
    xticks=[0,1/4,1/2,3/4,1],
    legend=:topleft)
plot!(x,y,label="wrapped",linetype=:scatter, marker=(2.5,stroke(0)))
plot!(x,yn,label="noisy",linetype=:scatter, marker=(2.5,stroke(0)))
savefig(string(resultsFolder,"original-wrapped-noisy",fileType))

meanSquaredError(MPow,yPow,ynPow)

# TV
α = 0.75
proxMaps = [ (λ,x) -> proxDistance(MPow,λ,ynPow,x), (λ,x) -> proxTV(MPow,α*λ,x) ]
costF = (x) -> L2TV(MPow,ynPow,α,x);
recTV = cyclicProximalPoint(MPow,costF,proxMaps,ynPow;
        debug = (d -> (d["Iteration"]%1000==1) ? print(d["Iteration"],"| λ=",d["λ"]," | last change ",distance(MPow,d["x"],d["xnew"]),"\n") : print("") ,
                Dict("λ"=>"","Iteration"=>0,"x"=>"","xnew"=>"") ,4),
        λ = i -> π/4/i
    );
meanSquaredError(MPow,yPow,recTV)
yRecTV = getValue.( getValue(recTV) )
plot(x,y,label="wrapped",linetype=:scatter, marker=(2.5,stroke(0)))
plot!(x,yn,label="noisy",linetype=:scatter, marker=(2.5,stroke(0)))
plot!(x,yRecTV,label="rec. (TV)",linetype=:scatter, marker=(2.5,stroke(0)))
savefig(string(resultsFolder,"wrapped-noisy-TV",fileType))

#
# TV2
β = 1
proxMaps2 = [ (λ,x) -> proxDistance(MPow,λ,yPow,x), (λ,x) -> proxTV2(MPow,β*λ,x) ]
costF2 = (x) -> L2TV2(MPow,yPow,β,x);
recTV2 = cyclicProximalPoint(MPow,costF2,proxMaps2,ynPow;
        debug = (d -> (d["Iteration"]%1000==1) ? print(d["Iteration"],"| λ=",d["λ"]," | last change ",distance(MPow,d["x"],d["xnew"]),"\n") : print("") ,
                Dict("λ"=>"","Iteration"=>0,"x"=>"","xnew"=>"") ,4),
        λ = i -> π/4/i
    );
meanSquaredError(MPow,yPow,recTV2)
yRecTV2 = getValue.( getValue(recTV2) )
plot(x,y,label="wrapped",linetype=:scatter, marker=(2.5,stroke(0)))
plot!(x,yn,label="noisy",linetype=:scatter, marker=(2.5,stroke(0)))
plot!(x,yRecTV2,label="rec. (TV2)",linetype=:scatter, marker=(2.5,stroke(0)))
savefig(string(resultsFolder,"wrapped-noisy-TV2",fileType))

#
# TV 1&2
α,β = 0.5,.5
proxMaps1p2 = [ (λ,x) -> proxDistance(MPow,λ,yPow,x), (λ,x) -> proxTV(MPow,α*λ,x), (λ,x) -> proxTV2(MPow,β*λ,x) ]
costF1p2 = (x) -> L2TVplusTV2(MPow,yPow,α,β,x);
recTV1p2 = cyclicProximalPoint(MPow,costF1p2,proxMaps1p2,ynPow;
        debug = (d -> (d["Iteration"]%1000==1) ? print(d["Iteration"],"| λ=",d["λ"]," | last change ",distance(MPow,d["x"],d["xnew"]),"\n") : print("") ,
                Dict("λ"=>"","Iteration"=>0,"x"=>"","xnew"=>"") ,4),
        λ = i -> π/4/i
    );
meanSquaredError(MPow,yPow,recTV1p2)
yRecTV1p2 = getValue.( getValue(recTV1p2) )
plot(x,y,label="wrapped",linetype=:scatter, marker=(2.5,stroke(0)))
plot!(x,yn,label="noisy",linetype=:scatter, marker=(2.5,stroke(0)))
plot!(x,yRecTV1p2,label="rec. (TV1&2)",linetype=:scatter, marker=(2.5,stroke(0)))
savefig(string(resultsFolder,"wrapped-noisy-TV1p2",fileType))

# TV on R
yRPow = PowPoint(RnPoint.(y))
ynRPow = PowPoint(RnPoint.(yn))
RPow = Power(Euclidean(1),size(yn))
α = 0.75
proxMapsR = [ (λ,x) -> proxDistance(RPow,λ,ynRPow,x), (λ,x) -> proxTV(RPow,α*λ,x) ]
costFR = (x) -> L2TV(RPow,ynRpow,α,x);
recTVR = cyclicProximalPoint(RPow,costFR,proxMapsR,ynRPow;
        debug = (d -> (d["Iteration"]%1000==1) ? print(d["Iteration"],"| λ=",d["λ"]," | last change ",distance(RPow,d["x"],d["xnew"]),"\n") : print("") ,
                Dict("λ"=>"","Iteration"=>0,"x"=>"","xnew"=>"") ,4),
        λ = i -> π/4/i
    );
yRecTVR = getValue.( getValue(recTVR) )
plot(x,y,label="wrapped",linetype=:scatter, marker=(2.5,stroke(0)))
plot!(x,yn,label="noisy",linetype=:scatter, marker=(2.5,stroke(0)))
plot!(x,yRecTVR,label="reconstructed",linetype=:scatter, marker=(2.5,stroke(0)))
savefig(string(resultsFolder,"wrapped-noisy-TVR",fileType))

# Export Data
df = DataFrame(x=x, y=y, yR=yR, yn=yn, yRecTV=yRecTV, yRecTV2=yRecTV2, yRecTV1p2=yRecTV1p2, yRecTVR=yRecTVR)
CSV.write(string(resultsFolder,"phase-data.csv"),df);
