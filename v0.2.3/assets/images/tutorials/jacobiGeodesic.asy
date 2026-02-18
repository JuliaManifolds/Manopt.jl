import settings;
import three;
import solids;unitsize(4cm);

currentprojection=perspective( camera = (1.0, 1.0, 0.5), target = (0.0, 0.0, 0.0) );
currentlight=nolight;

revolution S=sphere(O,1);
pen SpherePen = rgb(0.85,0.85,0.85)+opacity(0.6);
pen SphereLinePen = rgb(0.75,0.75,0.75)+opacity(0.6)+linewidth(0.5pt);
draw(surface(S), surfacepen=SpherePen, meshpen=SphereLinePen);

/*
  Colors
*/
pen curveStyle1 = rgb(0.0,0.0,0.0)+linewidth(0.75pt)+opacity(1.0);
pen pointStyle1 = rgb(0.9333333333333333,0.4666666666666667,0.2)+linewidth(3.5pt)+opacity(1.0);

/*
  Exported Points
*/
dot( (1.0,0.0,0.0), pointStyle1);
dot( (0.0,1.0,0.0), pointStyle1);

/*
  Exported Curves
*/
path3 p1 = (1.0,0.0,0.0) .. (0.9876883405951378,0.15643446504023087,0.0) .. (0.9510565162951535,0.3090169943749474,0.0) .. (0.8910065241883679,0.45399049973954675,0.0) .. (0.8090169943749475,0.5877852522924731,0.0) .. (0.7071067811865476,0.7071067811865475,0.0) .. (0.5877852522924731,0.8090169943749475,0.0) .. (0.45399049973954686,0.8910065241883678,0.0) .. (0.30901699437494745,0.9510565162951536,0.0) .. (0.15643446504023092,0.9876883405951378,0.0) .. (6.123233995736766e-17,1.0,0.0);
 draw(p1, curveStyle1);
