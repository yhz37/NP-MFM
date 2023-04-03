function Conc16 = FunctionofTripleT(deltaV12,deltaV34,deltaV56,Lambda12,Lambda34,Lambda56,Conc1,Conc2,Conc3,Conc4,Conc5,Conc6,nx,DetectorL,num,CG_Type,ratio)
%% Species Property
mu = 0.001;
Di = 1e-10;

%% Lentgth, width, aspect ratio, friction coefficient, perimeter and cross-sectional area of the channel of every channel of every channel
Ch(1).h = 60*1e-6;
Ch(1).L = 4000*1e-6;
Ch(1).w = 110.82*1e-6;

Ch(7).h = 60*1e-6;
Ch(7).L = 51239.342*1e-6;
Ch(7).w = 313.44465*1e-6;

Ch(10).h = 60*1e-6;
Ch(10).L = 10000*1e-6;
Ch(10).w = 1200*1e-6;
for i = [1,7,10]
    Ch(i).alpha = Ch(i).h / Ch(i).w;
    Ch(i).P = 2*(Ch(i).w+Ch(i).h);
    Ch(i).A = Ch(i).h*Ch(i).w;
    
%         Ch(i).K = 96*(1-1.3553*Ch(i).alpha+1.9467*Ch(i).alpha^2-1.7012*Ch(i).alpha^3+0.9564*Ch(i).alpha^4-0.2537*Ch(i).alpha^5);
%         Ch(i).R = Ch(i).K*Ch(i).L*Ch(i).P^2*mu/(32*Ch(i).A^3);
    index = 1:2:1001;
    Ch(i).R = 12/Ch(i).alpha*Ch(i).L*mu/(Ch(i).w^4*(1-192/Ch(i).alpha/pi^5*sum(tanh(index*pi/2*Ch(i).alpha)./index.^5)));
    
end

for i = 2:6
    Ch(i) = Ch(1);
end

for i = 8:9
    Ch(i) = Ch(7);
end

%% Voltage in the reservoir

V0 = 0;
Ch(10).V = Ch(10).R*(deltaV12/Ch(7).R+deltaV34/Ch(8).R+deltaV56/Ch(9).R)+V0;
Ch(7).V = Ch(10).V+deltaV12;
Ch(8).V = Ch(10).V+deltaV34;
Ch(9).V = Ch(10).V+deltaV56;
deltaV1 = deltaV12/Ch(7).R/(1/Ch(1).R+Lambda12/Ch(2).R);
Ch(1).V = Ch(7).V+deltaV1;
Ch(2).V = Ch(7).V+deltaV1*Lambda12;
deltaV3 = deltaV34/Ch(8).R/(1/Ch(3).R+Lambda34/Ch(4).R);
Ch(3).V = Ch(8).V+deltaV3;
Ch(4).V = Ch(8).V+deltaV3*Lambda34;
deltaV5 = deltaV56/Ch(9).R/(1/Ch(5).R+Lambda56/Ch(6).R);
Ch(5).V = Ch(9).V+deltaV5;
Ch(6).V = Ch(9).V+deltaV5*Lambda56;

%% Generate output matrix
Reserv1 = zeros(1,num+1);
Reserv2 = zeros(1,num+1);
Reserv3 = zeros(1,num+1);
Reserv4 = zeros(1,num+1);
Reserv5 = zeros(1,num+1);
Reserv6 = zeros(1,num+1);
Reserv1(1) = Conc1;
Reserv2(1) = Conc2;
Reserv3(1) = Conc3;
Reserv4(1) = Conc4;
Reserv5(1) = Conc5;
Reserv6(1) = Conc6;

%% Calculate the first upstream branch 1, top level

[s,Ch(7).Pe] = Combiner_Para_y(Ch(1),Ch(2),Ch(7),Ch(10).V,Di);
Am0 = Combiner_y(s,Reserv1,Reserv2,num);
Am = Channel_y(Am0,Ch(7),num);

%% Calculate the second upstream branch 2, top level

[s,Ch(8).Pe] = Combiner_Para_y(Ch(3),Ch(4),Ch(8),Ch(10).V,Di);
Bm0 = Combiner_y(s,Reserv3,Reserv4,num);
Bm = Channel_y(Bm0,Ch(8),num);

%% Calculate the third upstream branch 3, top level
[s,Ch(9).Pe] = Combiner_Para_y(Ch(5),Ch(6),Ch(9),Ch(10).V,Di);
Dm0 = Combiner_y(s,Reserv5,Reserv6,num);
Dm = Channel_y(Dm0,Ch(9),num);

%% Calculate the branch 14, then14-56 to obtain 16, bottom level
Q12 = (Ch(7).V-Ch(10).V)/Ch(7).R;
Q34 = (Ch(8).V-Ch(10).V)/Ch(8).R;
Q56 = (Ch(9).V-Ch(10).V)/Ch(9).R;
sa = Q12/(Q12+Q34);
sb = (Q12+Q34)/(Q12+Q34+Q56);

Fn0 = Combiner_y(sa,Am,Bm,num);
Gn0 = Combiner_y(sb,Fn0,Dm,num);

switch nargin
    case 16
        y7 = linspace(0,Ch(10).w,nx)';
    case 17
        y7_1 = linspace(0,Ch(10).w*ratio(1)/(ratio(1)+ratio(2)+ratio(3)),round(nx/3))';
        y7_2 = linspace(Ch(10).w*ratio(1)/(ratio(1)+ratio(2)+ratio(3)),Ch(10).w*(ratio(1)+ratio(2))/(ratio(1)+ratio(2)+ratio(3)),round(nx/3))';
        y7_3 = linspace(Ch(10).w*(ratio(1)+ratio(2))/(ratio(1)+ratio(2)+ratio(3)),Ch(10).w,nx-2*round(nx/3))';
        y7 = [y7_1;y7_2;y7_3];
end

Q16 = (Ch(10).V-V0)/Ch(10).R;
u16 = Q16/Ch(10).A;
Ch(10).Pe = u16*Ch(10).w/Di;

%% Define detection channel as Ch(11)
Ch(11) = Ch(10);
Ch(11).L = DetectorL*1e-6;
Gn = Channel_y(Gn0,Ch(11),num);
switch CG_Type
    case 'Coef'
        Conc16 = Gn';
        return
end
j = 1:num+1;
cosjpiy7 = cos((j-1).*pi.*(y7/Ch(11).w));
Conc16 = cosjpiy7*Gn';

end