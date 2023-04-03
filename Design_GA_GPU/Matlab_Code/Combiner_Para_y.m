function [s,Pe12] = Combiner_Para_y(Ch_Up1,Ch_Up2,Ch_Down,V0,Di)
Q1 = (Ch_Up1.V-Ch_Down.V)/Ch_Up1.R;
Q2 = (Ch_Up2.V-Ch_Down.V)/Ch_Up2.R;
s = Q1/(Q1+Q2);
Q12 = (Ch_Down.V-V0)/Ch_Down.R;
u12 = Q12/Ch_Down.A;
Pe12 = u12*Ch_Down.w/Di;
end