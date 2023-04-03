function [V1,V2,V3,V4,V5,V6] = deltaV2V(deltaV12,deltaV34,deltaV56,Lambda12,Lambda34,Lambda56)
%% Species Property
mu = 0.001;

%% Lentgth, width, aspect ratio, friction coefficient, perimeter and cross-sectional area of the channel of every channel  of every channel
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
V16 = Ch(10).R*(deltaV12/Ch(7).R+deltaV34/Ch(8).R+deltaV56/Ch(9).R)+V0;
V12 = V16+deltaV12;
V34 = V16+deltaV34;
V56 = V16+deltaV56;
deltaV1 = deltaV12/Ch(7).R/(1/Ch(1).R+Lambda12/Ch(2).R);
V1 = V12+deltaV1;
V2 = V12+deltaV1*Lambda12;
deltaV3 = deltaV34/Ch(8).R/(1/Ch(3).R+Lambda34/Ch(4).R);
V3 = V34+deltaV3;
V4 = V34+deltaV3*Lambda34;
deltaV5 = deltaV56/Ch(9).R/(1/Ch(5).R+Lambda56/Ch(6).R);
V5 = V56+deltaV5;
V6 = V56+deltaV5*Lambda56;
end