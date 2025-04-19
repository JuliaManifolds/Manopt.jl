import settings;
import three;
import solids;unitsize(4cm);

currentprojection=perspective( camera = (1.0, 0.5, 0.5), target = (0.0, 0.0, 0.0) );
currentlight=nolight;

revolution S=sphere(O,0.995);
pen SpherePen = rgb(0.85,0.85,0.85)+opacity(0.6);
pen SphereLinePen = rgb(0.75,0.75,0.75)+opacity(0.6)+linewidth(0.5pt);
draw(surface(S), surfacepen=SpherePen, meshpen=SphereLinePen);

/*
  Colors
*/
pen curveStyle1 = rgb(0.0,0.0,0.0)+linewidth(0.33pt)+opacity(1.0);
pen curveStyle2 = rgb(0.0,0.6,0.5333333333333333)+linewidth(0.66pt)+opacity(1.0);
pen curveStyle3 = rgb(0.0,0.4666666666666667,0.7333333333333333)+linewidth(0.33pt)+opacity(1.0);
pen pointStyle1 = rgb(0.0,0.4666666666666667,0.7333333333333333)+linewidth(3.5pt)+opacity(1.0);
pen pointStyle2 = rgb(0.9333333333333333,0.4666666666666667,0.2)+linewidth(3.5pt)+opacity(1.0);
pen pointStyle3 = rgb(0.0,0.6,0.5333333333333333)+linewidth(2.5pt)+opacity(1.0);
pen tVectorStyle1 = rgb(0.9333333333333333,0.4666666666666667,0.2)+linewidth(1.0pt)+opacity(1.0);

/*
  Exported Points
*/
dot( (0.8536738929290659,-0.4407611171876911,0.2774356179490873), pointStyle1);
dot( (0.8885045017775136,0.456669526292221,0.04486305915929584), pointStyle1);
dot( (0.9883816355291073,-0.06123674287166616,-0.13911076116273724), pointStyle1);
dot( (0.7807903227505065,0.40880133584733314,0.47249120595900573), pointStyle1);
dot( (0.5196380089577344,0.6019588838000035,0.6063182677939826), pointStyle1);
dot( (0.30442591317890266,0.5172990201008878,0.7998290987378815), pointStyle1);
dot( (0.4778114450918684,-0.008444328502415777,0.8784218327519892), pointStyle1);
dot( (0.8384410352962542,0.2421176388398039,0.48825780003394215), pointStyle2);
dot( (0.9943048600666006,-0.0867373475030919,0.06192316041730128), pointStyle3);
dot( (0.9348238168314646,0.11841135697995528,0.3347882644640029), pointStyle3);
dot( (0.9818430085961454,-0.17775320140764864,-0.06624277968354042), pointStyle3);
dot( (0.778531948021774,0.29683068677681756,0.552973371236831), pointStyle3);
dot( (0.5184843742829996,0.45490630566750234,0.7240401968732899), pointStyle3);
dot( (0.3080953048744573,0.5303076591937782,0.7898424334731149), pointStyle3);
dot( (0.6185257482733149,0.40501814600401254,0.673339587526539), pointStyle3);

/*
  Exported Curves
*/
path3 p1 = (0.4736801234111106,0.4730888976790247,0.7428418644490093) .. (0.48249802565690475,0.4695753170015591,0.7393880421673549) .. (0.4912666988522741,0.4660138257716564,0.7358587804656906) .. (0.4999852483335474,0.46240478736661994,0.7322544394329555) .. (0.508652784551153,0.45874857001496905,0.7285753868183988) .. (0.517268423160379,0.4550455467588693,0.724821997994057) .. (0.5258312851116025,0.4512960954160712,0.7209946559164562) .. (0.5343404967399784,0.447500598541361,0.7170937510875386) .. (0.5427951898545796,0.44365944338752983,0.71311968151482) .. (0.5511945018269773,0.43977302186586165,0.7090728526707809) .. (0.5595375756792552,0.43584173050614694,0.7049536774514968) .. (0.5678235601714466,0.4318659704162251,0.7007625761345105) .. (0.5760516098883857,0.4278461472410596,0.6964999763359514) .. (0.5842208853259647,0.42378267112134993,0.6921663129669052) .. (0.5923305529767888,0.41967595665168544,0.6877620281890418) .. (0.6003797854152179,0.41552642283824426,0.6832875713694999) .. (0.6083677613817888,0.4113344930560425,0.6787433990350399) .. (0.6162936658670076,0.40710059500573725,0.6741299748254634) .. (0.6241566901945049,0.40282516066998864,0.6694477694463091) .. (0.631956032103545,0.3985086262693846,0.6646972606208266) .. (0.6396908958308799,0.394151432217934,0.6598789330412344) .. (0.6473604921919403,0.3897540230781308,0.6549932783192671) .. (0.6549640386613562,0.3853168475155959,0.6500407949360167) .. (0.6625007594527977,0.38084035825330015,0.6450219881910726) .. (0.6699698855981282,0.37632501202537266,0.6399373701509659) .. (0.6773706550258607,0.371771269530501,0.6347874595969237) .. (0.6847023126389133,0.3671795953849259,0.6295727819719386) .. (0.6919641103916503,0.36255045807503716,0.6242938693271567) .. (0.6991553073662049,0.35788432990957364,0.6189512602675942) .. (0.7062751698480757,0.35318168697143426,0.6135454998971825) .. (0.7133229714009861,0.34844300906910325,0.6080771397631528) .. (0.720297992941003,0.3436687796876955,0.60254673779976) .. (0.7271995228099045,0.338859485939627,0.5969548582713594) .. (0.7340268568477896,0.33401561851491474,0.5913020717148331) .. (0.7407792984649234,0.32913767163111207,0.5855889548813793) .. (0.7474561587128108,0.32422614298288366,0.5798160906776666) .. (0.7540567563544883,0.31928153369122614,0.5739840681063598) .. (0.760580417934031,0.31430434825233894,0.568093482206025) .. (0.7670264778452647,0.30929509448615067,0.5621449339904174) .. (0.773394278399678,0.30425428348450645,0.5561390303871605) .. (0.7796831698935247,0.29918242955902175,0.5500763841758216) .. (0.7858925106741144,0.294080050188607,0.5439576139253899) .. (0.7920216672052784,0.2889476659666698,0.5377833439311643) .. (0.7980700141320106,0.28378580054799885,0.5315542041510571) .. (0.804036934344271,0.27859498059533566,0.5252708301413195) .. (0.8099218190399499,0.2733757357256395,0.5189338629916956) .. (0.8157240677869831,0.26812859845605075,0.5125439492600136) .. (0.8214430885846142,0.26285410414955857,0.5061017409062161) .. (0.8270782979237963,0.2575527909603777,0.4996078952258423) .. (0.8326291208467261,0.25222519977904123,0.49306307478296396) .. (0.8380949910055081,0.2468718741772137,0.48646794734258414) .. (0.8434753507199377,0.24149336035223065,0.47982318580250577) .. (0.8487696510344012,0.2360902070713702,0.47312946812467577) .. (0.8539773517738859,0.23066296561586297,0.4663874772660132) .. (0.8590979215990938,0.2252121897246448,0.45959790110872767) .. (0.8641308380606536,0.21973843553785916,0.45276143239013455) .. (0.8690755876524264,0.21424226154011436,0.44587876863197573) .. (0.8739316658638978,0.20872422850350164,0.4389506120692515) .. (0.8786985772316538,0.20318489943037998,0.431977669578572) .. (0.8833758353899317,0.1976248394959329,0.42496065260603494) .. (0.8879629631202449,0.19204461599050432,0.41790027709463684) .. (0.8924594924000724,0.1864447982617179,0.4107972634112262) .. (0.8968649644506113,0.18082595765638687,0.40365233627300423) .. (0.9011789297835852,0.1751886674622197,0.3964662246735827) .. (0.9054009482471066,0.16953350284932797,0.3892396618086049) .. (0.9095305890705839,0.1638610408115418,0.38197338500093786) .. (0.913567430908674,0.15817186010753975,0.37466813562544404) .. (0.9175110618842705,0.15246654120179798,0.3673246590333389) .. (0.921361079630529,0.1467456662053661,0.35994370447614304) .. (0.9251170913319195,0.14100981881647429,0.35252602502923674) .. (0.9287787137643048,0.13525958426097912,0.34507237751502345) .. (0.9323455733340413,0.129495549232653,0.3375835224257122) .. (0.935817306116097,0.12371830183332434,0.3300602238457244) .. (0.9391935578911814,0.11792843151287352,0.32250324937373515) .. (0.9424739841818874,0.11212652900909184,0.31491337004435493) .. (0.9456582502878375,0.10631318628740871,0.30729136024946185) .. (0.9487460313198334,0.10048899648049348,0.29963799765919036) .. (0.9517370122330047,0.09465455382773888,0.2919540631425864) .. (0.9546308878589516,0.0888104536146305,0.28424034068793497) .. (0.9574273629368825,0.08295729211201053,0.2764976173227711) .. (0.9601261521437384,0.07709566651524025,0.26872668303357916) .. (0.9627269801233046,0.07122617488326854,0.2609283306851906) .. (0.9652295815143052,0.06534941607761233,0.25310335593988964) .. (0.9676337009774774,0.05946598970125469,0.24525255717623043) .. (0.9699390932216242,0.05357649603746814,0.2373767354075805) .. (0.9721455230286413,0.04768153598856753,0.2294766942003926) .. (0.9742527652775155,0.04178171101460046,0.2215532395922176) .. (0.976260604967295,0.03587762307198064,0.2136071800094651) .. (0.9781688372390246,0.029969874552070508,0.2056393261849195) .. (0.9799772673966481,0.02405906821971951,0.19765049107502197) .. (0.9816857109268728,0.01814580715176428,0.18964148977692408) .. (0.9832939935179955,0.012230694675496895,0.18161313944532465) .. (0.984801951077687,0.006314334307107927,0.17356625920909508) .. (0.9862094297497352,0.0003973296901096657,0.16550167008770483) .. (0.9875162859297417,-0.0055197154662531,0.1574201949074523) .. (0.9887223862797749,-0.0114361974485995,0.14932265821751273) .. (0.9898276077419741,-0.017351512601008867,0.14120988620581015) .. (0.9908318375511043,-0.02326505738661183,0.13308270661472094) .. (0.9917349732460625,-0.0291762284491687,0.12494194865662034) .. (0.9925369226803318,-0.03508442267462947,0.11678844292927809);
 draw(p1, curveStyle1);
path3 p2 = (0.4766732571000727,0.4727843641994718,0.7411191206102499) .. (0.4855209881827112,0.4691333132333153,0.737687809611096) .. (0.4943185229916339,0.4654337602563841,0.7341802317149198) .. (0.5030649519812683,0.46168608775219255,0.7305967495576202) .. (0.5117593708896887,0.45789068317916026,0.7269377336225662) .. (0.5204008808321057,0.45404793893055445,0.7232035622022925) .. (0.5289885883937977,0.45015825229392153,0.7193946213593903) .. (0.5375216057224788,0.446222025410013,0.7155113048865931) .. (0.5459990506200895,0.4422396652312096,0.7115540142660639) .. (0.554420046634005,0.4382115834794481,0.7075231586278878) .. (0.5627837231476486,0.4341381966036548,0.7034191547077734) .. (0.5710892154705003,0.43001992573669,0.6992424268039672) .. (0.5793356649274953,0.4258571966518091,0.694993406733388) .. (0.5875222189477992,0.421650439718643,0.6906725337869826) .. (0.5956480311529516,0.41740008985870436,0.6862802546843083) .. (0.6037122614443706,0.4131065865004217,0.6818170235273497) .. (0.611714076090208,0.40877037353370926,0.6772833017535692) .. (0.6196526478115445,0.4043918992640745,0.6726795580882013) .. (0.6275271558679205,0.39997161636626927,0.6680062684957931) .. (0.6353367861421891,0.3955099818374897,0.6632639161309952) .. (0.6430807312246842,0.39100745695012873,0.6584529912886112) .. (0.6507581904966961,0.38646450720408687,0.6535739913529066) .. (0.6583683702132452,0.3818816022786454,0.6486274207461868) .. (0.6659104835851433,0.37725921598390905,0.6436137908766469) .. (0.6733837508603374,0.3725978262118189,0.6385336200854984) .. (0.6807873994045258,0.3678979148867464,0.6333874335933798) .. (0.6881206637810379,0.3631599679156676,0.6281757634460569) .. (0.6953827858299694,0.3583844751379281,0.622899148459415) .. (0.702573014746566,0.3535719302745994,0.6175581341637536) .. (0.7096906071588471,0.34872283087743544,0.6121532727473863) .. (0.7167348272044592,0.34383767827743233,0.606685122999551) .. (0.7237049466067551,0.33891697753299743,0.6011542502526394) .. (0.7306002447500869,0.33396123737773337,0.595561226323749) .. (0.7374200087543087,0.3289709701678416,0.5899066294555654) .. (0.7441635335484783,0.32394669182915226,0.5841910442565792) .. (0.7508301219437518,0.3188889218037837,0.5784150616406463) .. (0.7574190847054638,0.31379818299644,0.5725792787658941) .. (0.7639297406243849,0.3086750017203493,0.5666842989729852) .. (0.7703614165871491,0.30351990764285003,0.5607307317227389) .. (0.7767134476458448,0.2983334337306309,0.5547191925331214) .. (0.7829851770867616,0.29311611619462913,0.5486503029156099) .. (0.7891759564982854,0.28786849443459356,0.542524690310937) .. (0.795285145837935,0.2825911109833182,0.5363429880242211) .. (0.8013121134985344,0.27728451145055144,0.5301058351594918) .. (0.807256236373512,0.2719492444665875,0.5238138765536152) .. (0.8131168999213216,0.2665858616255459,0.5174677627096262) .. (0.8188934982289775,0.2611949174283433,0.5110681497294759) .. (0.8245854340746973,0.2557769692253665,0.5046156992461989) .. (0.8301921189896476,0.2503325771588496,0.4981110783555098) .. (0.8357129733187825,0.24486230410496262,0.4915549595468346) .. (0.8411474262807725,0.2393667156156184,0.4849480206337844) .. (0.846494916027016,0.23384637986000165,0.47829094468407857) .. (0.8517548896997261,0.2283018675658282,0.47158441994892486) .. (0.8569268034890889,0.22273375196033957,0.4648291397918635) .. (0.8620101226894861,0.21714260871103894,0.4580258026170833) .. (0.8670043217547759,0.21152901586617495,0.4511751117972154) .. (0.8719088843526284,0.20589355379497962,0.44427777560061477) .. (0.8767233034179057,0.20023680512766556,0.43733450711813426) .. (0.8814470812050872,0.19455935469519042,0.4303460241894013) .. (0.8860797293397286,0.18886178946879303,0.4233130493286028) .. (0.8906207688689534,0.18314469849930862,0.41623630964978675) .. (0.8950697303109705,0.177408672856269,0.40911653679168875) .. (0.8994261537036122,0.17165430556679395,0.40195446684209063) .. (0.9036895886518874,0.16588219155428027,0.3947508402617187) .. (0.9078595943745472,0.1600929275768949,0.3875064018076903) .. (0.9119357397496549,0.154287112165878,0.380221900456516) .. (0.9159176033591588,0.1484653455636633,0.3728980893266658) .. (0.9198047735324607,0.14262822966182076,0.3655357256007064) .. (0.9235968483889772,0.13677636793882952,0.3581355704470196) .. (0.9272934358796884,0.13091036539768625,0.35069838894110683) .. (0.9308941538276712,0.12503082850335606,0.3432249499864919) .. (0.9343986299676103,0.11913836512007227,0.33571602623522556) .. (0.937806501984286,0.11323358444849176,0.328172394008005) .. (0.941117417550032,0.1073170969627113,0.3205948332139119) .. (0.9443310343611625,0.10138951434715357,0.31298412726978103) .. (0.9474470201733606,0.09545144943332665,0.30534106301920516) .. (0.9504650528360279,0.08950351613646595,0.29766643065118636) .. (0.9533848203255912,0.08354632939206366,0.2899610236184412) .. (0.9562060207777614,0.07758050509229267,0.2822256385553684) .. (0.9589283625187413,0.07160666002233171,0.27446107519568785) .. (0.9615515640953817,0.0656254117965982,0.2666681362897588) .. (0.9640753543042793,0.059637378794895335,0.2588476275215862) .. (0.9664994722198157,0.05364318009847993,0.25100035742552484) .. (0.9688236672211332,0.04764343542605784,0.24312713730268654) .. (0.9710476990180461,0.04163876506971348,0.2352287811370637) .. (0.9731713376758832,0.03562978983078016,0.22730610551137395) .. (0.9751943636392596,0.029617130955657406,0.21935992952263667) .. (0.9771165677547765,0.023601410071582812,0.21139107469748947) .. (0.9789377512926438,0.017583249122364114,0.20340036490725355) .. (0.9806577259672272,0.011563270304078388,0.19538862628275655) .. (0.9822763139565134,0.0055420960007460385,0.187356687128922) .. (0.9837933479204951,-0.0004796512800158692,0.17930537783913358) .. (0.9852086710184713,-0.0065013489713521555,0.17123553080938364) .. (0.9865221369252628,-0.01252237451153454,0.1631479803522155) .. (0.9877336098463401,-0.018542105408325793,0.15504356261046587) .. (0.9888429645318627,-0.024559919303337563,0.14692311547081993) .. (0.9898500862896281,-0.03057519403637332,0.13878747847718442) .. (0.9907548709969297,-0.036587307709751216,0.13063749274389086) .. (0.9915572251113216,-0.042595638752600196,0.1224740008687355) .. (0.9922570656802892,-0.048599565985121856,0.11429784684586602);
 draw(p2, curveStyle2);
path3 p3 = (0.7807903227505065,0.40880133584733314,0.47249120595900573) .. (0.7808419209547756,0.4077057177217091,0.47335181653468056) .. (0.780892001442838,0.40660930714173704,0.4742115070602093) .. (0.7809405641173528,0.4055121062385016,0.47507027586461853) .. (0.780987608883929,0.4044141171446234,0.4759281212787261) .. (0.7810331356511262,0.4033153419942549,0.4767850416351446) .. (0.7810771443304543,0.40221578292307675,0.4776410352682849) .. (0.7811196348363739,0.4011154420682931,0.478496100514359) .. (0.7811606070862966,0.40001432156862765,0.47935023571138335) .. (0.781200061000585,0.39891242356431955,0.4802034391991823) .. (0.7812379965025531,0.3978097501971191,0.48105570931939096) .. (0.7812744135184655,0.39670630361028386,0.48190704441545873) .. (0.781309311977539,0.395602085948574,0.4827574428326524) .. (0.7813426918119416,0.3944970993582486,0.48360690291805924) .. (0.7813745529567934,0.3933913459870613,0.4844554230205906) .. (0.7814048953501659,0.3922848279842562,0.4853030014909847) .. (0.7814337189330829,0.3911775475005633,0.48614963668181) .. (0.7814610236495204,0.39006950668819496,0.48699532694746844) .. (0.7814868094464062,0.38896070770084124,0.4878400706441987) .. (0.781511076273621,0.38785115269366593,0.48868386613007914) .. (0.7815338240839972,0.3867408438233021,0.4895267117650311) .. (0.7815550528333202,0.38562978324784836,0.4903686059108224) .. (0.7815747624803278,0.3845179731268642,0.49120954693107005) .. (0.7815929529867106,0.38340541562136615,0.4920495331912437) .. (0.7816096243171119,0.3822921128938233,0.49288856305866885) .. (0.7816247764391275,0.3811780671081533,0.4937266349025297) .. (0.7816384093233066,0.38006328042971793,0.49456374709387285) .. (0.7816505229431512,0.37894775502531935,0.49539989800560996) .. (0.7816611172751158,0.37783149306319525,0.49623508601252103) .. (0.7816701922986083,0.37671449671301516,0.497069309491258) .. (0.78167774799599,0.3755967681458759,0.49790256682034734) .. (0.7816837843525747,0.3744783095342976,0.4987348563801933) .. (0.7816883013566296,0.37335912305221935,0.4995661765530814) .. (0.7816912989993751,0.3722392108749949,0.5003965257231813) .. (0.7816927772749845,0.37111857517938873,0.5012259022765498) .. (0.7816927361805849,0.3699972181435714,0.5020543046011345) .. (0.7816911757162557,0.36887514194711557,0.5028817310867764) .. (0.7816880958850303,0.36775234877099194,0.5037081801252131) .. (0.7816834966928949,0.3666288407975646,0.5045336501100822) .. (0.7816773781487887,0.3655046202105869,0.5053581394369244) .. (0.7816697402646045,0.3643796891951976,0.5061816465031864) .. (0.7816605830551878,0.363254049937916,0.507004169708224) .. (0.7816499065383377,0.3621277046266381,0.5078257074533055) .. (0.7816377107348056,0.3610006554506325,0.5086462581416145) .. (0.7816239956682968,0.3598729046005355,0.509465820178253) .. (0.781608761365469,0.35874445426834745,0.5102843919702451) .. (0.781592007855933,0.35761530664742824,0.5111019719265393) .. (0.7815737351722524,0.35648546393249314,0.5119185584580117) .. (0.7815539433499438,0.35535492831960835,0.5127341499774699) .. (0.7815326324274763,0.354223702006187,0.5135487448996551) .. (0.7815098024462718,0.3530917871909846,0.5143623416412455) .. (0.7814854534507049,0.35195918607409493,0.5151749386208597) .. (0.7814595854881022,0.35082590085694576,0.5159865342590596) .. (0.7814321986087435,0.3496919337422944,0.516797126978353) .. (0.7814032928658601,0.3485572869342239,0.5176067152031975) .. (0.7813728683156359,0.34742196263813796,0.5184152973600027) .. (0.781340925017207,0.34628596306075754,0.5192228718771341) .. (0.7813074630326614,0.34514929041011577,0.5200294371849155) .. (0.7812724824270385,0.34401194689555437,0.5208349917156324) .. (0.7812359832683303,0.3428739347277187,0.5216395339035347) .. (0.7811979656274797,0.3417352561185541,0.5224430621848405) .. (0.7811584295783812,0.340595913281301,0.523245574997738) .. (0.7811173751978808,0.3394559084304909,0.5240470707823897) .. (0.7810748025657758,0.3383152437819422,0.5248475479809345) .. (0.7810307117648138,0.33717392155275566,0.5256470050374912) .. (0.7809851028806942,0.3360319439613102,0.5264454403981617) .. (0.7809379760020662,0.33488931322725857,0.5272428525110332) .. (0.7808893312205303,0.3337460315715229,0.5280392398261826) .. (0.7808391686306367,0.3326021012162907,0.528834600795678) .. (0.7807874883298863,0.33145752438501025,0.5296289338735826) .. (0.7807342904187294,0.33031230330238637,0.5304222375159576) .. (0.7806795750005667,0.3291664401943762,0.5312145101808652) .. (0.7806233421817478,0.3280199372881847,0.532005750328371) .. (0.7805655920715722,0.32687279681226034,0.5327959564205481) .. (0.7805063247822881,0.32572502099629086,0.5335851269214792) .. (0.780445540429093,0.3245766120711992,0.5343732602972598) .. (0.780383239130133,0.3234275722691384,0.5351603550160015) .. (0.7803194210065025,0.32227790382348787,0.5359464095478345) .. (0.7802540861822445,0.32112760896884923,0.5367314223649108) .. (0.7801872347843495,0.3199766899410413,0.5375153919414073) .. (0.7801188669427562,0.3188251489770962,0.5382983167535288) .. (0.7800489827903507,0.3176729883152548,0.5390801952795102) .. (0.779977582462966,0.3165202101949628,0.5398610259996208) .. (0.7799046660993827,0.31536681685686563,0.540640807396166) .. (0.7798302338413275,0.31421281054280487,0.5414195379534911) .. (0.7797542858334738,0.3130581934958132,0.5421972161579834) .. (0.7796768222234413,0.3119029679601107,0.5429738404980763) .. (0.779597843161795,0.31074713618109995,0.5437494094642511) .. (0.7795173488020465,0.309590700405362,0.5445239215490407) .. (0.7794353393006515,0.3084336628806518,0.5452973752470321) .. (0.7793518148170114,0.307276025855894,0.5460697690548695) .. (0.7792667755134721,0.30611779158117824,0.5468411014712572) .. (0.779180221555324,0.30495896230775554,0.5476113709969627) .. (0.7790921531108009,0.30379954028803285,0.548380576134819) .. (0.779002570351081,0.3026395277755696,0.5491487153897284) .. (0.7789114734502857,0.3014789270250729,0.5499157872686649) .. (0.778818862585479,0.30031774029239305,0.5506817902806767) .. (0.7787247379366682,0.29915596983451953,0.55144672293689) .. (0.7786290996868025,0.2979936179095761,0.5522105837505114) .. (0.7785319480217734,0.29683068677681723,0.5529733712368305);
 draw(p3, curveStyle3);

/*
  Exported tangent vectors
*/
draw( (0.8384410352962542,0.2421176388398039,0.48825780003394215)--(1.376454196185712,-0.3019479653304221,-0.16583197004514216), tVectorStyle1,Arrow3(6.0));
