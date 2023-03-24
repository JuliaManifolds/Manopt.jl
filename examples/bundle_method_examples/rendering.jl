### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ 1c1f69c8-c330-11ed-2117-2ba0918ec216
using Pkg;
Pkg.activate();

# ╔═╡ e41fee0e-ad95-44c0-96ee-9457453f534e
using Random, LinearAlgebra, QuadraticModels, RipQP, Manifolds, Manopt, ColorSchemes

# ╔═╡ f8a8374a-b088-4739-babc-43f1867eb85f
begin
    Random.seed!(42)
    img = artificial_SPD_image(25, 1.5)
    M = SymmetricPositiveDefinite(3)
    N = PowerManifold(M, NestedPowerRepresentation(), size(img)[1], size(img)[2])
    f(N, q) = costTV2(N, q, 2)
    gradf(N, q) = grad_TV2(N, q)
end

# ╔═╡ bede3a2d-67c5-4925-81d3-ff306c9b2095
#s = subgradient_method(N, f, gradf, rand(N); stopping_criterion=StopAfterIteration(90), debug=[:Iteration, :Cost, "\n"])

# ╔═╡ 23668e76-8b34-43f6-af65-64eaaeba7dd6
#b = bundle_method(N, f, gradf, rand(N); stopping_criterion=StopAfterIteration(90), debug=[:Iteration, :Cost, "\n"])

# ╔═╡ fc28e3af-33ca-4f1b-9dbd-a90a0717348a
asymptote_export_SPD(
    "export1.fig"; data=img, scale_axes=(4.0, 4.0, 4.0), color_scheme=ColorSchemes.hsv
)

# ╔═╡ d5218834-ab32-46b4-bcd8-9789df00a787
render_asymptote("export1.fig")

# ╔═╡ b8311ec8-342c-42eb-b7b3-d8716a1af3af
#asymptote_export_SPD("export2.fig"; data=b, scale_axes=(6.,6.,6.))

# ╔═╡ 6dc21c0d-e2c1-41e9-8358-cf9b41b19102
#render_asymptote("export2.fig")

# ╔═╡ 2aa71666-c59f-416b-b11a-01931a08f695
img2 = map(p -> exp(M, p, rand(M; vector_at=p, tangent_distr=:Rician, σ=0.03)), img) #add (exp) noise to image

# ╔═╡ 997a9286-1788-46e6-8a5f-ae8cee7d43a2
asymptote_export_SPD(
    "export2.fig"; data=img2, scale_axes=(12.0, 12.0, 12.0), color_scheme=ColorSchemes.hsv
)

# ╔═╡ 7c03e397-1dd8-4f58-9bb9-c412284b8690
render_asymptote("export2.fig")

# ╔═╡ eee8994b-7167-4e06-9f61-a4bd54c399f3
#s = subgradient_method(N, f, gradf, img2; stopping_criterion=StopAfterIteration(90), debug=[:Iteration, :Cost, "\n"])

# ╔═╡ d1cd19f4-1412-4831-bed3-8b1abf25bfa0
begin
    img3 = artificial_SPD_image2(16)
    asymptote_export_SPD(
        "export3.fig"; data=img3, scale_axes=(8.0, 8.0, 8.0), color_scheme=ColorSchemes.hsv
    )
    render_asymptote("export3.fig")
end

# ╔═╡ 97f8f8b5-553c-4e19-9dd4-9e80feda37c9
function normal_cone_vector(M, p)
    Y = rand(M; vector_at=p)
    (norm(M, p, Y) > 1.0) && (Y /= norm(M, p, Y))
    return Y
end

# ╔═╡ ec0de758-b00b-4bdd-a0d3-b2361b03c74a
begin
    data = img3#map(p -> exp(M, p, rand(M; vector_at=p, tangent_distr=:Rician, σ=0.03)), img3)
    α = 6.0
    L = PowerManifold(M, NestedPowerRepresentation(), size(img3)[1], size(img3)[2])
    g(L, q) = 1 / (2 * α) * distance(L, data, q)^2 + costTV(L, q)
    function gradg(L, q)
        if q ≈ data
            #println(norm(L, q, 1/α * normal_cone_vector(L, q) + grad_TV(L, q)))
            return 1 / α * normal_cone_vector(L, q) + grad_TV(L, q)
        else
            #println(norm(L, q, 1/α * grad_distance(L, data, q) + grad_TV(L, q)))
            return 1 / α * grad_distance(L, data, q) + grad_TV(L, q)
        end
    end
end

# ╔═╡ 6749f982-99cb-4dd2-ad56-9aec03a3663d
s = subgradient_method(
    L,
    g,
    gradg,
    data;
    stepsize=ConstantStepsize(1e-2),
    debug=[
        #:Iteration, 
        :Stop,
        #(:Cost,"F(p): %1.6e"), 
        #"\n"
    ],
    #stopping_criterion=StopWhenCostLess(38.74)
)

# ╔═╡ d9416139-6170-421a-b181-641d50c98695
b = bundle_method(
    L,
    g,
    gradg,
    data;
    m=1e-8,
    diam=0.1,
    debug=[:Iteration, (:Cost, "F(p): %1.15e "), :Stop, "\n"],
    stopping_criterion=StopAfterIteration(100),
)

# ╔═╡ ed110367-2d6c-4eba-adb1-14ae9a4b1175
g(L, s)

# ╔═╡ a9d0b9a7-1ed3-4f1e-aefe-8e518f1899bf
begin
    asymptote_export_SPD(
        "b.fig"; data=s, scale_axes=(8.0, 8.0, 8.0), color_scheme=ColorSchemes.hsv
    )
    render_asymptote("b.fig")
end

# ╔═╡ ab48f6e5-d803-4872-845c-dfb92a61440a
md"""b=bundle_method(L, g, gradg, data; 
m=1e-3, diam=.1, debug=[:Iteration, (:Cost,"F(p): %1.15e "), :Stop, "\n"], stopping_criterion=StopAfterIteration(1000)
)
	Initial F(p): 4.272619178099957e+01 
	# 1     F(p): 4.272619178099957e+01 
	# 2     F(p): 4.272619178099957e+01 
	# 3     F(p): 4.272619178099957e+01 
	# 4     F(p): 4.272619178099957e+01 
	# 5     F(p): 4.272619178099957e+01 
	# 6     F(p): 4.272619178099957e+01 
	# 7     F(p): 4.272619178099957e+01 
	# 8     F(p): 4.272619178099957e+01 
	stepped in
	# 9     F(p): 3.883334647446237e+01 
	# 10    F(p): 4.677644313113185e+01 
	# 11    F(p): 8.650729566019500e+01 
	# 12    F(p): 8.757686632912815e+01 
	# 13    F(p): 1.002702156196126e+02 
	# 14    F(p): 1.308169745508361e+02 
	# 15    F(p): 1.610032533607690e+02 
	# 16    F(p): 1.666579743081804e+02 
	# 17    F(p): 1.529470721930266e+02 
	# 18    F(p): 1.244403710487932e+02 
	# 19    F(p): 9.277531609180205e+01 
	# 20    F(p): 7.531585336281094e+01 
	# 21    F(p): 7.079098345861118e+01 
	# 22    F(p): 7.429453898805885e+01 
	# 23    F(p): 8.442320924856676e+01 
	# 24    F(p): 9.275150533387769e+01 
	# 25    F(p): 9.186066260374912e+01 
	# 26    F(p): 8.401745541959326e+01 
	# 27    F(p): 7.699102811489189e+01 
	# 28    F(p): 7.498999487101796e+01 
	# 29    F(p): 7.670378503267014e+01 
	# 30    F(p): 7.935672420673998e+01 
	# 31    F(p): 7.747203717749886e+01 
	# 32    F(p): 7.248911471725987e+01 
	# 33    F(p): 6.811776295315306e+01 
	# 34    F(p): 6.730287606597503e+01 
	# 35    F(p): 6.568699319470969e+01 
	# 36    F(p): 6.475141694004293e+01 
	# 37    F(p): 6.246542486131198e+01 
	# 38    F(p): 5.860832161288369e+01 
	# 39    F(p): 5.637821823127302e+01 
	# 40    F(p): 5.570161005668005e+01 
	# 41    F(p): 5.550838614901822e+01 
	# 42    F(p): 5.818692693162699e+01 
	# 43    F(p): 5.914702461168763e+01 
	# 44    F(p): 6.156219492600120e+01 
	# 45    F(p): 6.393379777413373e+01 
	# 46    F(p): 6.614142469782162e+01 
	# 47    F(p): 6.191337722901088e+01 
	# 48    F(p): 5.571065611902470e+01 
	# 49    F(p): 5.124870988235940e+01 
	# 50    F(p): 4.830126483260644e+01 
	# 51    F(p): 4.788055886215940e+01 
	# 52    F(p): 4.640288113910205e+01 
	# 53    F(p): 4.527534467765513e+01 
	# 54    F(p): 4.639337268754088e+01 
	# 55    F(p): 4.675511915921170e+01 
	# 56    F(p): 4.745260304356087e+01 
	# 57    F(p): 4.821502000792537e+01 
	# 58    F(p): 4.947035731626074e+01 
	# 59    F(p): 5.090488745124918e+01 
	# 60    F(p): 5.185501427416440e+01 
	# 61    F(p): 5.206397954354880e+01 
	# 62    F(p): 5.376746439235738e+01 
	# 63    F(p): 5.245307967803599e+01 
	# 64    F(p): 4.964624528646903e+01 
	# 65    F(p): 4.832909640297480e+01 
	# 66    F(p): 4.626249832846980e+01 
	# 67    F(p): 4.461981687461520e+01 
	# 68    F(p): 4.201819933579750e+01 
	# 69    F(p): 4.171158248639617e+01 
	# 70    F(p): 3.951306699180463e+01 
	# 71    F(p): 3.917272558153967e+01 
	# 72    F(p): 3.873317910941165e+01 
	# 73    F(p): 3.906753421494115e+01 
	# 74    F(p): 3.916205699570985e+01 
	# 75    F(p): 3.942286657056617e+01 
	# 76    F(p): 3.951352469415178e+01 
	# 77    F(p): 3.932392072232395e+01 
	# 78    F(p): 4.101801020491908e+01 
	# 79    F(p): 4.417836241087786e+01 
	# 80    F(p): 4.632006359089084e+01 
	# 81    F(p): 4.915670430159170e+01 
	# 82    F(p): 5.038059392944914e+01 
	# 83    F(p): 4.946481432993563e+01 
	# 84    F(p): 4.511668900538452e+01 
	# 85    F(p): 4.113983952112321e+01 
	# 86    F(p): 3.840545814614642e+01 
	# 87    F(p): 3.730820291625943e+01 
	# 88    F(p): 3.897198200802153e+01 
	# 89    F(p): 3.906992234011034e+01 
	# 90    F(p): 3.956790171459595e+01 
	# 91    F(p): 4.106360519094491e+01 
	# 92    F(p): 4.089467743001084e+01 
	# 93    F(p): 4.056937375931534e+01 
	# 94    F(p): 4.111563463593002e+01 
	# 95    F(p): 4.071392586039335e+01 
	# 96    F(p): 3.919080345715087e+01 
	# 97    F(p): 3.744078350376050e+01 
	# 98    F(p): 3.797693605916933e+01 
	# 99    F(p): 3.782738519936385e+01 
	# 100   F(p): 3.757657901357031e+01 
	# 101   F(p): 3.802167071523400e+01 
	# 102   F(p): 3.834036047984813e+01 
	# 103   F(p): 3.921453622291930e+01 
	# 104   F(p): 3.730910278169192e+01 
	# 105   F(p): 3.488668759796685e+01 
	# 106   F(p): 3.370495704807580e+01 
	# 107   F(p): 3.492917496919354e+01 
	# 108   F(p): 3.617410525689008e+01 
	# 109   F(p): 3.873241136569477e+01 
	# 110   F(p): 4.040864721184711e+01 
	# 111   F(p): 4.043714820283825e+01 
	# 112   F(p): 3.866394904468430e+01 
	# 113   F(p): 3.631072401417907e+01 
	# 114   F(p): 3.463179145161079e+01 
	# 115   F(p): 3.505913496896765e+01 
	# 116   F(p): 3.608396521898295e+01 
	# 117   F(p): 3.614107412509237e+01 
	# 118   F(p): 3.676850978306226e+01 
	# 119   F(p): 3.754458561798264e+01 
	# 120   F(p): 3.697313402631144e+01 
	# 121   F(p): 3.547697590379983e+01 
	# 122   F(p): 3.608231691010021e+01 
	# 123   F(p): 3.656687416197171e+01 
	# 124   F(p): 3.646335985803392e+01 
	# 125   F(p): 3.566756194740757e+01 
	# 126   F(p): 3.432907397400033e+01 
	# 127   F(p): 3.448513919085239e+01 
	# 128   F(p): 3.485836767725507e+01 
	# 129   F(p): 3.586827192399025e+01 
	# 130   F(p): 3.614342615955587e+01 
	# 131   F(p): 3.615236852638205e+01 
	# 132   F(p): 3.500693745636106e+01 
	# 133   F(p): 3.491192362530771e+01 
	# 134   F(p): 3.497877507975243e+01 
	# 135   F(p): 3.494630964475888e+01 
	# 136   F(p): 3.568994696111173e+01 
	# 137   F(p): 3.507861107929622e+01 
	# 138   F(p): 3.551212198432683e+01 
	# 139   F(p): 3.677524900494678e+01 
	# 140   F(p): 3.571842690145557e+01 
	# 141   F(p): 3.450594960967046e+01 
	# 142   F(p): 3.311398568920556e+01 
	# 143   F(p): 3.125970489134610e+01 
	# 144   F(p): 3.149427928516100e+01 
	# 145   F(p): 3.275384755806949e+01 
	# 146   F(p): 3.389008947559721e+01 
	# 147   F(p): 3.500461920971670e+01 
	# 148   F(p): 3.444361968814020e+01 
	# 149   F(p): 3.392378458026592e+01 
	# 150   F(p): 3.311119693054331e+01 
	# 151   F(p): 3.356881319267782e+01 
	# 152   F(p): 3.451739176323680e+01 
	# 153   F(p): 3.445156091718974e+01 
	# 154   F(p): 3.348842238518228e+01 
	# 155   F(p): 3.400365840707428e+01 
	# 156   F(p): 3.408334005075486e+01 
	# 157   F(p): 3.276108374447868e+01 
	# 158   F(p): 3.259661994349604e+01 
	# 159   F(p): 3.189492191025216e+01 
	# 160   F(p): 3.094742882499403e+01 
	# 161   F(p): 3.054779906245086e+01 
	# 162   F(p): 3.111808750519015e+01 
	# 163   F(p): 3.157353845921613e+01 
	# 164   F(p): 3.213803746680053e+01 
	# 165   F(p): 3.288769577468124e+01 
	# 166   F(p): 3.426581194294245e+01 
	# 167   F(p): 3.612939578600535e+01 
	# 168   F(p): 3.603792862745381e+01 
	# 169   F(p): 3.553577011348556e+01 
	# 170   F(p): 3.394859920351805e+01 
	# 171   F(p): 3.413945103028443e+01 
	# 172   F(p): 3.379661207160501e+01 
	# 173   F(p): 3.346387752500222e+01 
	# 174   F(p): 3.407114531429625e+01 
	# 175   F(p): 3.365574280468711e+01 
	# 176   F(p): 3.321797044634606e+01 
	# 177   F(p): 3.214867276033491e+01 
	# 178   F(p): 3.150474983588115e+01 
	# 179   F(p): 3.161732362083272e+01 
	# 180   F(p): 3.128473888181017e+01 
	# 181   F(p): 3.186515750311501e+01 
	# 182   F(p): 3.327040125854220e+01 
	# 183   F(p): 3.341173818842326e+01 
	# 184   F(p): 3.315134333795123e+01 
	# 185   F(p): 3.202762812953257e+01 
	# 186   F(p): 3.072259051369343e+01 
	# 187   F(p): 3.019127122226299e+01 
	# 188   F(p): 3.054609552128101e+01 
	# 189   F(p): 3.152587334156673e+01 
	# 190   F(p): 3.129278571924745e+01 
	# 191   F(p): 3.191539300347280e+01 
	# 192   F(p): 3.199375968383231e+01 
	# 193   F(p): 3.250343968221060e+01 
	# 194   F(p): 3.258276116459241e+01 
	# 195   F(p): 3.220930141493028e+01 
	# 196   F(p): 3.250247840908904e+01 
	# 197   F(p): 3.174421890695337e+01 
	# 198   F(p): 3.181834205182035e+01 
	# 199   F(p): 3.107419967116675e+01 
	# 200   F(p): 3.047705700965076e+01 
	# 201   F(p): 3.073504933462354e+01 
	# 202   F(p): 3.169217628429401e+01 
	# 203   F(p): 3.207890378850699e+01 
	# 204   F(p): 3.211168226048495e+01 
	# 205   F(p): 3.141714799251871e+01 
	# 206   F(p): 3.144408525181854e+01 
	# 207   F(p): 3.292858749320913e+01 
	# 208   F(p): 3.426227083405620e+01 
	# 209   F(p): 3.261253049554564e+01 
	# 210   F(p): 3.154168765803053e+01 
	# 211   F(p): 3.141863316068367e+01 
	# 212   F(p): 3.179918250964450e+01 
	# 213   F(p): 3.095007083236576e+01 
	# 214   F(p): 2.916039040589602e+01 
	# 215   F(p): 2.898960683931561e+01 
	# 216   F(p): 2.861905837849558e+01 
	# 217   F(p): 2.923352151816768e+01 
	# 218   F(p): 3.135996373719695e+01 
	# 219   F(p): 3.207289123789339e+01 
	# 220   F(p): 3.181057013217365e+01 
	# 221   F(p): 3.147306897454590e+01 
	# 222   F(p): 3.243521973338751e+01 
	# 223   F(p): 3.294819638642741e+01 
	# 224   F(p): 3.271633611038931e+01 
	# 225   F(p): 3.274505313932833e+01 
	# 226   F(p): 3.204008519847485e+01 
	# 227   F(p): 3.023535078262645e+01 
	# 228   F(p): 3.012802147272481e+01 
	# 229   F(p): 3.008320620065965e+01 
	# 230   F(p): 3.095321745289277e+01 
	# 231   F(p): 2.991644188717644e+01 
	# 232   F(p): 3.118796609094604e+01 
	# 233   F(p): 3.176570433368512e+01 
	# 234   F(p): 3.169627776738212e+01 
	# 235   F(p): 3.138956738976441e+01 
	# 236   F(p): 3.142794470225968e+01 
	# 237   F(p): 3.143282027172503e+01 
	# 238   F(p): 3.196970217284315e+01 
	# 239   F(p): 3.305986983573467e+01 
	# 240   F(p): 3.270525876916253e+01 
	# 241   F(p): 3.130046717948355e+01 
	# 242   F(p): 3.069266931270700e+01 
	# 243   F(p): 2.917916561446332e+01 
	# 244   F(p): 2.870892885551779e+01 
	# 245   F(p): 2.915496344360037e+01 
	# 246   F(p): 2.982054891165679e+01 
	# 247   F(p): 3.099869092954615e+01 
	# 248   F(p): 3.168240849128849e+01 
	# 249   F(p): 3.059319429089548e+01 
	# 250   F(p): 2.949455688256864e+01 
	# 251   F(p): 3.037243470901641e+01 
	# 252   F(p): 3.068525691388854e+01 
	# 253   F(p): 3.015103468076283e+01 
	# 254   F(p): 3.011159125423832e+01 
	# 255   F(p): 3.114728189726384e+01 
	# 256   F(p): 3.018073404882549e+01 
	# 257   F(p): 2.965789940606668e+01 
	# 258   F(p): 3.066881248423456e+01 
	# 259   F(p): 3.016614074346689e+01 
	# 260   F(p): 3.036198043340759e+01 
	# 261   F(p): 3.033964692635916e+01 
	# 262   F(p): 3.025777006677816e+01 
	# 263   F(p): 2.992196879766311e+01 
	# 264   F(p): 2.961860628705816e+01 
	# 265   F(p): 2.932758978566954e+01 
	# 266   F(p): 2.997171976384085e+01 
	# 267   F(p): 2.958522512275224e+01 
	# 268   F(p): 3.002905598142445e+01 
	# 269   F(p): 2.883641317675256e+01 
	# 270   F(p): 2.801085068701055e+01 
	# 271   F(p): 2.778663173539758e+01 
	# 272   F(p): 2.704024811564495e+01 
	# 273   F(p): 2.687043248948848e+01 
	# 274   F(p): 2.839640493087071e+01 
	# 275   F(p): 2.893832423103439e+01 
	# 276   F(p): 2.913466250500439e+01 
	# 277   F(p): 2.872081313587660e+01 
	# 278   F(p): 2.921526012315799e+01 
	# 279   F(p): 2.863055110189190e+01 
	# 280   F(p): 2.830842296376390e+01 
	# 281   F(p): 2.862302998900346e+01 
	# 282   F(p): 2.818059736960399e+01 
	# 283   F(p): 2.852065431106721e+01 
	# 284   F(p): 2.876644491455552e+01 
	# 285   F(p): 3.000669109290924e+01 
	# 286   F(p): 3.004464064254638e+01 
	# 287   F(p): 2.860714513733663e+01 
	# 288   F(p): 2.819839560099398e+01 
	# 289   F(p): 2.801689370730633e+01 
	# 290   F(p): 2.778844313659954e+01 
	# 291   F(p): 2.733372292639407e+01 
	# 292   F(p): 2.808409820313049e+01 
	# 293   F(p): 2.813066664892636e+01 
	# 294   F(p): 2.810957191763906e+01 
	# 295   F(p): 2.804431290666441e+01 
	# 296   F(p): 2.848316762305937e+01 
	# 297   F(p): 2.873584075867337e+01 
	# 298   F(p): 2.695663038646380e+01 
	# 299   F(p): 2.650506862529392e+01 
	# 300   F(p): 2.645734227060117e+01 
	# 301   F(p): 2.624141562286370e+01 
	# 302   F(p): 2.795713601269324e+01 
	# 303   F(p): 2.765630772551397e+01 
	# 304   F(p): 2.814669470419346e+01 
	# 305   F(p): 2.870782473967643e+01 
	# 306   F(p): 2.825142516933560e+01 
	# 307   F(p): 2.823915571265935e+01 
	# 308   F(p): 2.853318997789989e+01 
	# 309   F(p): 2.764041288573475e+01 
	# 310   F(p): 2.754756064257973e+01 
	# 311   F(p): 2.726686525266992e+01 
	# 312   F(p): 2.702104727536122e+01 
	# 313   F(p): 2.730844975695315e+01 
	# 314   F(p): 2.821970114181249e+01 
	# 315   F(p): 2.788419679393664e+01 
	# 316   F(p): 2.702479940651658e+01 
	# 317   F(p): 2.764674124758968e+01 
	# 318   F(p): 2.749413415589747e+01 
	# 319   F(p): 2.681480232309261e+01 
	# 320   F(p): 2.625826561580184e+01 
	# 321   F(p): 2.658452115794735e+01 
	# 322   F(p): 2.657616124750170e+01 
	# 323   F(p): 2.750503203471310e+01 
	# 324   F(p): 2.705698119630744e+01 
	# 325   F(p): 2.690651285563849e+01 
	# 326   F(p): 2.668309264328341e+01 
	# 327   F(p): 2.632921259719777e+01 
	# 328   F(p): 2.582216984394706e+01 
	# 329   F(p): 2.565396779825680e+01 
	# 330   F(p): 2.607978764685098e+01 
	# 331   F(p): 2.673794696365690e+01 
	# 332   F(p): 2.664875780942094e+01 
	# 333   F(p): 2.674453890709124e+01 
	# 334   F(p): 2.778812246549698e+01 
	# 335   F(p): 2.738889612378413e+01 
	# 336   F(p): 2.639601307535175e+01 
	# 337   F(p): 2.599126257910634e+01 
	# 338   F(p): 2.605651885696364e+01 
	# 339   F(p): 2.599349974572994e+01 
	# 340   F(p): 2.622573132881876e+01 
	# 341   F(p): 2.627853981196961e+01 
	# 342   F(p): 2.634036268799861e+01 
	# 343   F(p): 2.684168143176080e+01 
	# 344   F(p): 2.662407936683381e+01 
	# 345   F(p): 2.625534072002543e+01 
	# 346   F(p): 2.622764457584974e+01 
	# 347   F(p): 2.641441491766312e+01 
	# 348   F(p): 2.713685948918899e+01 
	# 349   F(p): 2.756656003979394e+01 
	# 350   F(p): 2.663739367495360e+01 
	# 351   F(p): 2.643518718596873e+01 
	# 352   F(p): 2.749079574984585e+01 
	# 353   F(p): 2.814684293235548e+01 
	# 354   F(p): 2.671314867053335e+01 
	# 355   F(p): 2.630972201305470e+01 
	# 356   F(p): 2.620682639705286e+01 
	# 357   F(p): 2.562775245398736e+01 
	# 358   F(p): 2.597489774487277e+01 
	# 359   F(p): 2.584321681016641e+01 
	# 360   F(p): 2.528325124905858e+01 
	# 361   F(p): 2.599248447970063e+01 
	# 362   F(p): 2.645852197357621e+01 
	# 363   F(p): 2.694198662380659e+01 
	# 364   F(p): 2.728914571351501e+01 
	# 365   F(p): 2.734008060042213e+01 
	# 366   F(p): 2.683942631046505e+01 
	# 367   F(p): 2.657582672124066e+01 
	# 368   F(p): 2.623774467594350e+01 
	# 369   F(p): 2.651205037653121e+01 
	# 370   F(p): 2.633841392606761e+01 
	# 371   F(p): 2.608808659666855e+01 
	# 372   F(p): 2.617037962799095e+01 
	# 373   F(p): 2.498574494112325e+01 
	# 374   F(p): 2.469103895304040e+01 
	# 375   F(p): 2.572596831558082e+01 
	# 376   F(p): 2.561125511042085e+01 
	# 377   F(p): 2.579099671004715e+01 
	# 378   F(p): 2.562879894718269e+01 
	# 379   F(p): 2.614714916870625e+01 
	# 380   F(p): 2.538001999826558e+01 
	# 381   F(p): 2.570030377550709e+01 
	# 382   F(p): 2.493406197499834e+01 
	# 383   F(p): 2.510123535920857e+01 
	# 384   F(p): 2.436727494226432e+01 
	# 385   F(p): 2.358811457753468e+01 
	# 386   F(p): 2.370344906217721e+01 
	# 387   F(p): 2.488675192808273e+01 
	# 388   F(p): 2.586174540758593e+01 
	# 389   F(p): 2.504988893188790e+01 
	# 390   F(p): 2.535425563801113e+01 
	# 391   F(p): 2.573863129711215e+01 
	# 392   F(p): 2.663594584845226e+01 
	# 393   F(p): 2.689640673403712e+01 
	# 394   F(p): 2.575956743949058e+01 
	# 395   F(p): 2.622029536099040e+01 
	# 396   F(p): 2.680877436663066e+01 
	# 397   F(p): 2.669847631215594e+01 
	# 398   F(p): 2.688623661704735e+01 
	# 399   F(p): 2.591314647597554e+01 
	# 400   F(p): 2.534159400620593e+01 
	# 401   F(p): 2.543996333246134e+01 
	# 402   F(p): 2.578897468832375e+01 
	# 403   F(p): 2.517965186455528e+01 
	# 404   F(p): 2.449884913669657e+01 
	# 405   F(p): 2.517312181350099e+01 
	# 406   F(p): 2.526873560790570e+01 
	# 407   F(p): 2.592967327846942e+01 
	# 408   F(p): 2.601745002046660e+01 
	# 409   F(p): 2.614439240212687e+01 
	# 410   F(p): 2.613508898491614e+01 
	# 411   F(p): 2.529462156191683e+01 
	# 412   F(p): 2.510876376922422e+01 
	# 413   F(p): 2.457766883779180e+01 
	# 414   F(p): 2.427183201481563e+01 
	# 415   F(p): 2.473629675650821e+01 
	# 416   F(p): 2.465419622156685e+01 
	# 417   F(p): 2.474586580566695e+01 
	# 418   F(p): 2.529966471876249e+01 
	# 419   F(p): 2.521129436652193e+01 
	# 420   F(p): 2.573411209754052e+01 
	# 421   F(p): 2.587247171528443e+01 
	# 422   F(p): 2.547086270137749e+01 
	# 423   F(p): 2.484194158410863e+01 
	# 424   F(p): 2.455150760619693e+01 
	# 425   F(p): 2.469346967263333e+01 
	# 426   F(p): 2.428465867453849e+01 
	# 427   F(p): 2.434543661792323e+01 
	# 428   F(p): 2.452928964234908e+01 
	# 429   F(p): 2.424728105285902e+01 
	# 430   F(p): 2.418549325353094e+01 
	# 431   F(p): 2.425820121107992e+01 
	# 432   F(p): 2.463111257691035e+01 
	# 433   F(p): 2.454172514079595e+01 
	# 434   F(p): 2.467744641000245e+01 
	# 435   F(p): 2.402833526127823e+01 
	# 436   F(p): 2.404317328361109e+01 
	# 437   F(p): 2.469187176776364e+01 
	# 438   F(p): 2.488123761178531e+01 
	# 439   F(p): 2.495744575524714e+01 
	# 440   F(p): 2.441762968509900e+01 
	# 441   F(p): 2.344438340180695e+01 
	# 442   F(p): 2.343773396717287e+01 
	# 443   F(p): 2.407758127812092e+01 
	# 444   F(p): 2.363105740482763e+01 
	# 445   F(p): 2.365308538989755e+01 
	# 446   F(p): 2.380374839295485e+01 
	# 447   F(p): 2.447818853201994e+01 
	# 448   F(p): 2.524144120911255e+01 
	# 449   F(p): 2.543334691260641e+01 
	# 450   F(p): 2.499542901681349e+01 
	# 451   F(p): 2.415344076002184e+01 
	# 452   F(p): 2.344227038060902e+01 
	# 453   F(p): 2.307259198488430e+01 
	# 454   F(p): 2.319998760660055e+01 
	# 455   F(p): 2.349942075318131e+01 
	# 456   F(p): 2.373938739479569e+01 
	# 457   F(p): 2.400384091095345e+01 
	# 458   F(p): 2.414145169743219e+01 
	# 459   F(p): 2.394442155725317e+01 
	# 460   F(p): 2.379420058103245e+01 
	# 461   F(p): 2.463663673204192e+01 
	# 462   F(p): 2.448381609638140e+01 
	# 463   F(p): 2.441680768676145e+01 
	# 464   F(p): 2.362153803740901e+01 
	# 465   F(p): 2.370089068488394e+01 
	# 466   F(p): 2.397377731214124e+01 
	# 467   F(p): 2.394961794474934e+01 
	# 468   F(p): 2.348332401282216e+01 
	# 469   F(p): 2.376683424153083e+01 
	# 470   F(p): 2.352377732836313e+01 
	# 471   F(p): 2.313590504204062e+01 
	# 472   F(p): 2.292982972498584e+01 
	# 473   F(p): 2.335964074205618e+01 
	# 474   F(p): 2.366200523409924e+01 
	# 475   F(p): 2.477862823031160e+01 
	# 476   F(p): 2.522207054547970e+01 
	# 477   F(p): 2.524514331123972e+01 
	# 478   F(p): 2.536548077965534e+01 
	# 479   F(p): 2.412513244810091e+01 
	# 480   F(p): 2.363939599145389e+01 
	# 481   F(p): 2.307536296065693e+01 
	# 482   F(p): 2.379485526718260e+01 
	# 483   F(p): 2.395946187451752e+01 
	# 484   F(p): 2.387686815397991e+01 
	# 485   F(p): 2.361253649879685e+01 
	# 486   F(p): 2.298908026286995e+01 
	# 487   F(p): 2.358044112166616e+01 
	# 488   F(p): 2.402056315181699e+01 
	# 489   F(p): 2.336327869978857e+01 
	# 490   F(p): 2.346263242175603e+01 
	# 491   F(p): 2.323445255464109e+01 
	# 492   F(p): 2.307317104859967e+01 
	# 493   F(p): 2.287411292646117e+01 
	# 494   F(p): 2.352589541317690e+01 
	# 495   F(p): 2.386790252536293e+01 
	# 496   F(p): 2.372919397174343e+01 
	# 497   F(p): 2.360880113199560e+01 
	# 498   F(p): 2.377349983126650e+01 
	# 499   F(p): 2.306927962713474e+01 
	# 500   F(p): 2.335045880199770e+01 
	# 501   F(p): 2.309520475717681e+01 
	# 502   F(p): 2.262140417236137e+01 
	# 503   F(p): 2.278654939178769e+01 
	# 504   F(p): 2.350620445443672e+01 
	# 505   F(p): 2.392841151073244e+01 
	# 506   F(p): 2.272456118564060e+01 
	# 507   F(p): 2.242467996019769e+01 
	# 508   F(p): 2.236620910662836e+01 
	# 509   F(p): 2.207067498975970e+01 
	# 510   F(p): 2.240035583867530e+01 
	# 511   F(p): 2.304268877516286e+01 
	# 512   F(p): 2.295754777984016e+01 
	# 513   F(p): 2.319664214253581e+01 
	# 514   F(p): 2.305643707814317e+01 
	# 515   F(p): 2.285468532144817e+01 
	# 516   F(p): 2.348334973915583e+01 
	# 517   F(p): 2.340107621188194e+01 
	# 518   F(p): 2.298228584954045e+01 
	# 519   F(p): 2.305292502714032e+01 
	# 520   F(p): 2.301161617822078e+01 
	# 521   F(p): 2.303785226282243e+01 
	# 522   F(p): 2.291613592086749e+01 
	# 523   F(p): 2.259867290254113e+01 
	# 524   F(p): 2.290748555381987e+01 
	# 525   F(p): 2.287954848143202e+01 
	# 526   F(p): 2.263496259122887e+01 
	# 527   F(p): 2.246009377532110e+01 
	# 528   F(p): 2.222024533560525e+01 
	# 529   F(p): 2.169769178276020e+01 
	# 530   F(p): 2.308810868223453e+01 
	# 531   F(p): 2.374384601049138e+01 
	# 532   F(p): 2.345624246945809e+01 
	# 533   F(p): 2.398767379572986e+01 
	# 534   F(p): 2.397248771235830e+01 
	# 535   F(p): 2.432342312172870e+01 
	# 536   F(p): 2.429529044470291e+01 
	# 537   F(p): 2.324114650160826e+01 
	# 538   F(p): 2.226660769475071e+01 
	# 539   F(p): 2.246749064176041e+01 
	# 540   F(p): 2.325449418812756e+01 
	# 541   F(p): 2.331056910711850e+01 
	# 542   F(p): 2.289986951268737e+01 
	# 543   F(p): 2.264331801149114e+01 
	# 544   F(p): 2.268177732193865e+01 
	# 545   F(p): 2.312565420125120e+01 
	# 546   F(p): 2.257918880118743e+01 
	# 547   F(p): 2.293757531554034e+01 
	# 548   F(p): 2.219755945173178e+01 
	# 549   F(p): 2.272280165893685e+01 
	# 550   F(p): 2.257893013202540e+01 
	# 551   F(p): 2.276473610415027e+01 
	# 552   F(p): 2.289409459858258e+01 
	# 553   F(p): 2.251525982528795e+01 
	# 554   F(p): 2.271175312440210e+01 
	# 555   F(p): 2.306492464217241e+01 
	# 556   F(p): 2.278674976133054e+01 
	# 557   F(p): 2.285392458437336e+01 
	# 558   F(p): 2.259097493699538e+01 
	# 559   F(p): 2.205755198628316e+01 
	# 560   F(p): 2.234615696626800e+01 
	# 561   F(p): 2.239661618475439e+01 
	# 562   F(p): 2.286802191055438e+01 
	# 563   F(p): 2.283950386232969e+01 
	# 564   F(p): 2.173823849701669e+01 
	# 565   F(p): 2.129893567006308e+01 
	# 566   F(p): 2.115253215220934e+01 
	# 567   F(p): 2.155108534000580e+01 
	# 568   F(p): 2.198563824496204e+01 
	# 569   F(p): 2.236793454956774e+01 
	# 570   F(p): 2.285208872883888e+01 
	# 571   F(p): 2.244489282034974e+01 
	# 572   F(p): 2.246601159564882e+01 
	# 573   F(p): 2.187585703631771e+01 
	# 574   F(p): 2.164731499251682e+01 
	# 575   F(p): 2.166414044038869e+01 
	# 576   F(p): 2.195010951900642e+01 
	# 577   F(p): 2.265647131340849e+01 
	# 578   F(p): 2.177381210941661e+01 
	# 579   F(p): 2.097890392288402e+01 
	# 580   F(p): 2.153249652985851e+01 
	# 581   F(p): 2.245624855583151e+01 
	# 582   F(p): 2.253256981512646e+01 
	# 583   F(p): 2.291246857462901e+01 
	# 584   F(p): 2.315409657004904e+01 
	# 585   F(p): 2.192269457595959e+01 
	# 586   F(p): 2.142715974750890e+01 
	# 587   F(p): 2.176664899497587e+01 
	# 588   F(p): 2.217212770638950e+01 
	# 589   F(p): 2.243928091174963e+01 
	# 590   F(p): 2.209200599470311e+01 
	# 591   F(p): 2.192630947506000e+01 
	# 592   F(p): 2.214691612063262e+01 
	# 593   F(p): 2.315838966881118e+01 
	# 594   F(p): 2.325461472679590e+01 
	# 595   F(p): 2.316222119967827e+01 
	# 596   F(p): 2.304395907826971e+01 
	# 597   F(p): 2.269084621883666e+01 
	# 598   F(p): 2.290396261906129e+01 
	# 599   F(p): 2.275175809973537e+01 
	# 600   F(p): 2.241401989832639e+01 
	# 601   F(p): 2.190974208109221e+01 
	# 602   F(p): 2.194162307786225e+01 
	# 603   F(p): 2.251653388161913e+01 
	# 604   F(p): 2.192462013076054e+01 
	# 605   F(p): 2.151541986744694e+01 
	# 606   F(p): 2.170254652379244e+01 
	# 607   F(p): 2.228435965106717e+01 
	# 608   F(p): 2.282194198416362e+01 
	# 609   F(p): 2.164914219180416e+01 
	# 610   F(p): 2.224142523501551e+01 
	# 611   F(p): 2.262568565036072e+01 
	# 612   F(p): 2.240747933946594e+01 
	# 613   F(p): 2.195899169991274e+01 
	# 614   F(p): 2.204718911380931e+01
"""

# ╔═╡ Cell order:
# ╠═1c1f69c8-c330-11ed-2117-2ba0918ec216
# ╠═e41fee0e-ad95-44c0-96ee-9457453f534e
# ╠═f8a8374a-b088-4739-babc-43f1867eb85f
# ╠═bede3a2d-67c5-4925-81d3-ff306c9b2095
# ╠═23668e76-8b34-43f6-af65-64eaaeba7dd6
# ╠═fc28e3af-33ca-4f1b-9dbd-a90a0717348a
# ╠═d5218834-ab32-46b4-bcd8-9789df00a787
# ╠═b8311ec8-342c-42eb-b7b3-d8716a1af3af
# ╠═6dc21c0d-e2c1-41e9-8358-cf9b41b19102
# ╠═2aa71666-c59f-416b-b11a-01931a08f695
# ╠═997a9286-1788-46e6-8a5f-ae8cee7d43a2
# ╠═7c03e397-1dd8-4f58-9bb9-c412284b8690
# ╠═eee8994b-7167-4e06-9f61-a4bd54c399f3
# ╠═d1cd19f4-1412-4831-bed3-8b1abf25bfa0
# ╠═97f8f8b5-553c-4e19-9dd4-9e80feda37c9
# ╠═ec0de758-b00b-4bdd-a0d3-b2361b03c74a
# ╠═6749f982-99cb-4dd2-ad56-9aec03a3663d
# ╠═d9416139-6170-421a-b181-641d50c98695
# ╠═ed110367-2d6c-4eba-adb1-14ae9a4b1175
# ╠═a9d0b9a7-1ed3-4f1e-aefe-8e518f1899bf
# ╠═ab48f6e5-d803-4872-845c-dfb92a61440a
