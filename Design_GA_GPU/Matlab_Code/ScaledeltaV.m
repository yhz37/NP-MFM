function deltaV = ScaledeltaV(x)
a = 0.013732653608351371142440565461532;
c = 50;
lowbound = 0.366025403784439;
upbound = 0.999867159008185;
deltaV = log(1./((upbound-lowbound).*x+lowbound)-1)./(-a)+c;
end
