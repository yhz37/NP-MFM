function dn = Combiner_y(s,ReservL,ReservR,num)
dn = zeros(1,num+1);
dn(1) = ReservL(1)*s+ReservR(1)*(1-s);

i = 1:num;
m = (0:num)';

%% Part 1
f1 = m-i*s;
f2 = m+i*s;
part1 = ReservL'.*s.*(f1.*sin(f2.*pi)+f2.*sin(f1.*pi))./(f1.*f2.*pi);

[ii,jj] = find(abs(f1) <= 1e-13);
part1(abs(f1) <= 1e-13) = ReservL(ii)'.*(sin(2*jj*pi*s)./(2*jj*pi)+s);
part1 = sum(part1);

%% Part 2
F1 = pi*(m+i-i*s);
F2 = pi*(m-i+i*s);
part2 = 2.*(-1).^i.*(1-s).*ReservR'.*(cos(F2./2).*sin(F1./2)./F1+cos(F1./2).*sin(F2./2)./F2);
    
[ii,jj] = find(abs(F2) <= 1e-13);

part2(abs(F2) <= 1e-13) = ReservR(ii)'.*((-1).^(jj+1).*sin(jj.*pi.*(s-1))./jj./pi+cos(jj.*pi.*s).*(1-s));

part2 = sum(part2);

%% Sum 

dn(2:num+1) = part1+part2;

end