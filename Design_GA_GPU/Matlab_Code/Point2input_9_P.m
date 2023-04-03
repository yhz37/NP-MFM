function input = Point2input_9_P(point)
deltaV = ScaledeltaV(point(1:3));
[V1,V2,V3,V4,V5,V6] = deltaV2V(deltaV(1),deltaV(2),deltaV(3),1,1,1);
input = [V1 point(4) V2 point(5) V3 point(6) V4 point(7) V5 point(8) V6 point(9)];
end
