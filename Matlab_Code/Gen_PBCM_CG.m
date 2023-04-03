function output_CG = Gen_PBCM_CG(input,ROM_Config)

Num = size(input,1);

if ~isfield(ROM_Config,'CG_Type')
    output_CG = zeros(ROM_Config.m,Num);
else
    switch ROM_Config.CG_Type
        case 'Coef'
            if ~isfield(ROM_Config,'FS_Num')
                ROM_Config.FS_Num = 700;
            end
            output_CG = zeros(ROM_Config.FS_Num+1,Num);
        case 'CG'
            output_CG = zeros(ROM_Config.m,Num);
    end    
end


switch ROM_Config.ProbType
    case 'mCGG_6'
        mCGG_input = [ROM_Config.DefP.*ones(ROM_Config.x_n,3) input];
    case {'mCGG_7P','mCGG_1P_BatchC'}
        mCGG_input = [ROM_Config.DefP.*input(:,1) input(:,2:end)];
    case 'mCGG_3P'
        mCGG_input = [ROM_Config.DefP.*input(:,1) input(:,2) input(:,3) input(:,2) input(:,3) input(:,2) input(:,3)];
    case {'mCGG_9','mCGG_3P_BatchC'}
        mCGG_input = input;
end

parr = gcp('nocreate');
if isempty(parr)
    parpool(10);
end
for i = 1:Num
    output_CG(:,i) = CG_9(mCGG_input(i,:),ROM_Config);
end
end