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
dot( (0.49245597062127383,-0.5932250990343978,-0.6368446426524194), pointStyle1);
dot( (0.9979022426575619,-0.05689072048806097,0.030895954773364043), pointStyle1);
dot( (0.9320102169461956,0.36102760793532584,0.031875097119887175), pointStyle1);
dot( (0.643151699404565,-0.0256990356251639,0.7653074226224104), pointStyle1);
dot( (-0.0561944327895707,-0.43625871968158,0.8980648725037892), pointStyle1);
dot( (-0.11053770045126271,0.7568039089266176,0.6442276462651525), pointStyle1);
dot( (0.022811923298921316,0.0045484193598182565,0.9997294274136034), pointStyle1);
