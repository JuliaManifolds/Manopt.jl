import settings;
import three;
import solids;unitsize(4cm);

currentprojection=perspective( camera = (1.2, 1.0, 0.5), target = (0.0, 0.0, 0.0) );
currentlight=nolight;

revolution S=sphere(O,1);
draw(surface(S), surfacepen=lightgrey+opacity(.6), meshpen=0.6*white+linewidth(.5pt));

/*
  Colors
*/
pen pointStyle1 = rgb(0.0,0.0,0.0)+linewidth(3.5pt)+opacity(1.0);
pen pointStyle2 = rgb(0.0,0.4666666666666667,0.7333333333333333)+linewidth(3.5pt)+opacity(1.0);
pen tVectorStyle1 = rgb(0.2,0.7333333333333333,0.9333333333333333)+opacity(1.0);

/*
  Exported Points
*/
dot( (1.0,0.0,0.0), pointStyle1);
dot( (0.11061587104123714,0.11061587104123714,0.9876883405951378), pointStyle1);
dot( (0.0,1.0,0.0), pointStyle1);
dot( (0.7071067811865476,0.7071067811865475,0.0), pointStyle2);
dot( (-0.7071067811865476,-0.7071067811865475,-0.0), pointStyle2);

/*
  Exported tangent vectors
*/
draw( (1.0,0.0,0.0)--(1.0,4.967699558375198e-18,0.7071067811865475), tVectorStyle1,Arrow3(6.0));
draw( (0.11061587104123714,0.11061587104123714,0.9876883405951378)--(0.8090169943749476,0.8090169943749475,0.8312538755549069), tVectorStyle1,Arrow3(6.0));
draw( (0.0,1.0,0.0)--(-4.9676995583751935e-18,1.0,0.7071067811865475), tVectorStyle1,Arrow3(6.0));
