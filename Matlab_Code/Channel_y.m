function Am = Channel_y(Am0,Ch_Down,num)
j = 1:num+1;
Am = Am0.*exp(-((j-1).*pi).^2.*Ch_Down.L./(Ch_Down.Pe*Ch_Down.w));
end

