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
pen pointStyle1 = rgb(0.0,0.4666666666666667,0.7333333333333333)+linewidth(3.5pt)+opacity(1.0);
pen pointStyle2 = rgb(0.0,0.6,0.5333333333333333)+linewidth(3.5pt)+opacity(1.0);
pen pointStyle3 = rgb(0.9333333333333333,0.4666666666666667,0.2)+linewidth(3.5pt)+opacity(1.0);
pen pointStyle4 = rgb(0.9333333333333333,0.2,0.4666666666666667)+linewidth(3.5pt)+opacity(1.0);

/*
  Exported Points
*/
dot( (0.7071067811865475,0.0,0.7071067811865475), pointStyle1);
dot( (0.9013817292525768,-0.3266930346539438,0.2842228690275139), pointStyle2);
dot( (0.5413128376734956,-0.27073756886516653,0.7960411927625647), pointStyle2);
dot( (0.11640703550675724,0.22835279004025902,0.9665942299462364), pointStyle2);
dot( (0.8120710958917794,0.3048940575972932,0.4975742646670772), pointStyle2);
dot( (0.34037597269891057,0.16278898828946248,0.9260906772562629), pointStyle2);
dot( (0.7937947657628919,-0.11512297480761367,0.5971905646599591), pointStyle2);
dot( (0.8657115032997768,0.0994828021701859,0.49055760632856193), pointStyle2);
dot( (0.9937317675245106,0.10346338978158666,0.042337940284278175), pointStyle2);
dot( (0.5984410921632728,-0.236358593721975,0.7655082457976526), pointStyle2);
dot( (0.9634889828946449,-0.2406208001795652,0.1174334294891714), pointStyle2);
dot( (0.8673667369419585,-0.14699434441962997,0.4754656731619039), pointStyle2);
dot( (0.6209143934751815,-0.015857957553400644,0.7837179602111859), pointStyle2);
dot( (0.11593369432252859,0.34579073997729104,0.9311219805517899), pointStyle2);
dot( (0.8597873278395911,0.43766600080255863,0.26308596052991867), pointStyle2);
dot( (0.9835484300817445,-0.013809287592808289,0.18011604387148494), pointStyle2);
dot( (0.3602751592138037,0.35779950254465337,0.861499463512466), pointStyle2);
dot( (0.3758750750898673,-0.4355379560131802,0.8179392500657049), pointStyle2);
dot( (0.7757170268296428,-0.0364912940107325,0.6300249834316106), pointStyle2);
dot( (0.6536438240556595,-0.3249314023609702,0.6834978676145428), pointStyle2);
dot( (0.39855796029336676,-0.0887956300863908,0.9128345350414564), pointStyle2);
dot( (0.6002317749655175,-0.5086143481254278,0.6172789168622966), pointStyle2);
dot( (0.6534977497781589,-0.04623041645882359,0.7555153470505588), pointStyle2);
dot( (0.9788538215436897,-0.18315567225496213,0.091099921899787), pointStyle2);
dot( (0.9609566690166504,-0.17153168680353964,0.2171155468748286), pointStyle2);
dot( (0.9072110142857421,-0.07013917089536566,0.41478750254165775), pointStyle2);
dot( (0.7195716535495816,-0.06411437829420373,0.6914520821458765), pointStyle2);
dot( (-0.1849041908329384,0.5602200342452995,0.807442848406382), pointStyle2);
dot( (0.234113963290437,0.2556971851368949,0.937981663842909), pointStyle2);
dot( (0.7148289545793283,0.2936163156762542,0.6346723760049168), pointStyle2);
dot( (0.8921550973858754,-0.25406027973916584,0.37351393075393563), pointStyle2);
dot( (0.4482158074375948,0.08515137838890918,0.8898605692587651), pointStyle2);
dot( (0.8280902756156101,0.3858233801200921,0.40670236633632895), pointStyle2);
dot( (0.0728662106242115,0.08193007748052097,0.9939708133307054), pointStyle2);
dot( (0.6400965064494377,-0.2601437995326629,0.7229119351594181), pointStyle2);
dot( (0.7035801723463732,-0.1003489204851328,0.7034948722190635), pointStyle2);
dot( (0.5596910124493797,-0.02369135209940644,0.8283626563402592), pointStyle2);
dot( (0.2096532978644124,-0.8419747445620077,0.49711570505699954), pointStyle2);
dot( (0.7273627283578351,0.602011625766135,0.3294320321982385), pointStyle2);
dot( (0.6681220780684642,0.2181027485334902,0.7113677529095719), pointStyle2);
dot( (0.8096550939643867,-0.30889835793427434,0.4990395107434172), pointStyle2);
dot( (0.7198603589536396,-0.34913132647125744,0.5999236455446286), pointStyle2);
dot( (0.6223361503684609,0.6258299122188214,0.4701432089446222), pointStyle2);
dot( (0.7825683141616745,-0.29558732713148916,0.5479187583113181), pointStyle2);
dot( (0.23695782718487035,0.24265954061185052,0.940727024957756), pointStyle2);
dot( (0.4974143877693706,-0.5624392947814862,0.6604854022048681), pointStyle2);
dot( (0.44317099909010615,-0.21190138583909196,0.8710323003453713), pointStyle2);
dot( (0.5255983249736583,-0.153539840650367,0.8367627609531536), pointStyle2);
dot( (0.452627705575377,-0.25846818514792586,0.8534180437581029), pointStyle2);
dot( (0.6509927511437905,0.3476947382992546,0.6747716702094506), pointStyle2);
dot( (0.6884804858344034,0.5820170078263757,0.4327248816812552), pointStyle2);
dot( (0.3912302831709517,-0.062434887924444106,0.9181725057416166), pointStyle2);
dot( (0.5400128999795748,-0.4162610281845638,0.731514062934117), pointStyle2);
dot( (0.24898546625025264,0.31176134470202493,0.916957524395612), pointStyle2);
dot( (0.839880820298772,0.13133834785232457,0.5266407182108914), pointStyle2);
dot( (0.6375333356550568,-0.686624382418478,0.3494255334073057), pointStyle2);
dot( (0.40814696811155443,0.2894877365255597,0.8658018842799164), pointStyle2);
dot( (0.25946600412328236,0.3510054637935926,0.8997069284446695), pointStyle2);
dot( (0.19956388081481158,0.3423932136267531,0.9181182629359228), pointStyle2);
dot( (0.8352835857313747,-0.06558196186170072,0.545894071854703), pointStyle2);
dot( (0.1807808473326897,-0.9462130902662805,0.268326430018383), pointStyle2);
dot( (0.914444707018359,-0.12422510751084297,0.38517398727072283), pointStyle2);
dot( (0.6946294245385879,-0.4748943223988742,0.5403381766250692), pointStyle2);
dot( (0.456291972020519,-0.3053882255340298,0.8357844626306488), pointStyle2);
dot( (0.6298237524955624,-0.06415129621874843,0.774084395906456), pointStyle2);
dot( (0.1916775227159408,0.17526202246132297,0.9656826346001293), pointStyle2);
dot( (0.11144017741329243,-0.3516536196218806,0.9294734093372028), pointStyle2);
dot( (0.970910321044485,-0.23730955695847497,0.031895496005975965), pointStyle2);
dot( (0.2332864877006272,-0.005139710229540168,0.9723944662713075), pointStyle2);
dot( (0.8570088090528865,0.49400849375260625,-0.14659982710096914), pointStyle2);
dot( (0.6738898415286123,-0.30984624932931815,0.670721837471463), pointStyle2);
dot( (0.205059952786316,-0.3620413011854729,0.9093275053572303), pointStyle2);
dot( (0.36105729775221895,-0.4180548583894944,0.8335872858416238), pointStyle2);
dot( (0.4624294636629965,0.058548921918656856,0.8847208683412617), pointStyle2);
dot( (0.14758954352099912,0.1377490095549117,0.9794092796221117), pointStyle2);
dot( (0.9659404885344638,-0.10819257007919808,0.23506029096693298), pointStyle2);
dot( (0.9999706868520573,0.005760229604234521,0.005044322702942694), pointStyle2);
dot( (0.7708706411928757,-0.2699107517909207,0.5769806241240206), pointStyle2);
dot( (0.776922121343302,0.06800088015216309,0.6259136503272275), pointStyle2);
dot( (0.38950666856157273,0.09889241879535356,0.9156991015889824), pointStyle2);
dot( (0.8503339262256128,0.07897307266878843,0.5202840259925191), pointStyle2);
dot( (0.7025236746717578,0.3427317444330536,0.6236949878615077), pointStyle2);
dot( (0.7701635233994575,0.41084447185120127,0.4879087693147568), pointStyle2);
dot( (0.6019974338521028,0.6833377386428,0.4130963865516464), pointStyle2);
dot( (0.9233787435159421,0.37773408806752473,0.06847375215887935), pointStyle2);
dot( (0.4448847434083096,0.15099946985409285,0.8827665179345585), pointStyle2);
dot( (0.8052571625752776,0.1854538212250568,0.5631765107976694), pointStyle2);
dot( (0.5210389499912665,0.3514092009897261,0.7778367348304918), pointStyle2);
dot( (0.8168863033456829,-0.5099412570457227,0.2695490340714522), pointStyle2);
dot( (0.6836813099517501,0.12099044228175662,0.7196813039805354), pointStyle2);
dot( (0.9572639618455621,-0.07067283445623072,0.2804479592039502), pointStyle2);
dot( (0.5592866060041981,0.30365663188048897,0.7713566893852042), pointStyle2);
dot( (0.6355248508478041,-0.42674253944714435,0.6434275164935667), pointStyle2);
dot( (0.25569537183572244,-0.32934463142726106,0.9089290349481817), pointStyle2);
dot( (0.46591190831530616,0.4231266977207518,0.7771035268005937), pointStyle2);
dot( (-0.24540230556351394,0.6192543288278474,0.7458564102104935), pointStyle2);
dot( (0.7206649443671707,0.026606223795364103,0.6927727959552205), pointStyle2);
dot( (-0.1556170638176626,0.4709844443817593,0.8683069633483166), pointStyle2);
dot( (0.864645918606244,0.15287339377544437,0.47855737473488213), pointStyle2);
dot( (0.8283553683993905,0.539804408097735,0.14979514225155605), pointStyle2);
dot( (0.7766227187182986,-0.11478608022900104,0.6194201389656778), pointStyle2);
dot( (0.6868392769979477,0.00653160266211623,0.7267799844104132), pointStyle3);
dot( (0.6895661616463938,-0.020194930269146705,0.7239410704634467), pointStyle4);
