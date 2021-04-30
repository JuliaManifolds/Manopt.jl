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
dot( (0.8536738929290659,-0.4407611171876911,0.2774356179490873), pointStyle1);
dot( (0.8885045017775136,0.456669526292221,0.04486305915929584), pointStyle1);
dot( (0.9883816355291073,-0.06123674287166616,-0.13911076116273724), pointStyle1);
dot( (0.7807903227505065,0.40880133584733314,0.47249120595900573), pointStyle1);
dot( (0.5196380089577344,0.6019588838000035,0.6063182677939826), pointStyle1);
dot( (0.30442591317890266,0.5172990201008878,0.7998290987378815), pointStyle1);
dot( (0.4778114450918684,-0.008444328502415777,0.8784218327519892), pointStyle1);
