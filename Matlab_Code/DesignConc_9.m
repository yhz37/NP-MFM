function DesignConc = DesignConc_9(Point,m)

N1section = round(m*Point(1)/(Point(1)+Point(2)+Point(3)));
N2section = round(m*Point(2)/(Point(1)+Point(2)+Point(3)));
N3section = m-N1section-N2section;
y1 = (linspace(Point(4),Point(5),N1section))';
y2 = (linspace(Point(6),Point(7),N2section))';
y3 = (linspace(Point(8),Point(9),N3section))';
DesignConc = [y1;y2;y3];

end