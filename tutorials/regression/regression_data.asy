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

/*
  Exported Points
*/
dot( (0.9837117687894378,-0.17755131744105618,0.0280479165019622), pointStyle1);
dot( (0.9838420262712789,-0.05982681995903209,-0.1687472042909554), pointStyle1);
dot( (-0.0917525024594581,0.20000514884698506,0.9754893227130282), pointStyle1);
dot( (0.07844742532720095,0.5643971870536215,0.8217674955274712), pointStyle1);
dot( (0.8640095918651447,-0.32042627848327054,0.3883483297535702), pointStyle1);
dot( (0.09891843208768877,-0.3385482932829229,0.9357351104391312), pointStyle1);
dot( (0.5391307236127518,0.6639024159319173,0.518238984423744), pointStyle1);
