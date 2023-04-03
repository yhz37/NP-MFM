function ConcPred = CG_9(TriTErrinput,TriTConfig)
deltaV = ScaledeltaV(TriTErrinput(1:3));                       % scale function from normalized pressure to total pressure
if ~isfield(TriTConfig,'DetectorL')
    TriTConfig.DetectorL = 400;                                % Detector location(mm)
end
if ~isfield(TriTConfig,'FS_Num')
    TriTConfig.FS_Num = 700;                                   % Number of FS
end

if ~isfield(TriTConfig,'CG_Type')
    if TriTConfig.DP == 1
        ConcPred = FunctionofTripleT(deltaV(1),deltaV(2),deltaV(3),1,1,1,TriTErrinput(4),TriTErrinput(5),TriTErrinput(6),TriTErrinput(7),TriTErrinput(8),TriTErrinput(9),TriTConfig.m,TriTConfig.DetectorL,TriTConfig.FS_Num,'CG',TriTConfig.ratio);
    else
        ConcPred = FunctionofTripleT(deltaV(1),deltaV(2),deltaV(3),1,1,1,TriTErrinput(4),TriTErrinput(5),TriTErrinput(6),TriTErrinput(7),TriTErrinput(8),TriTErrinput(9),TriTConfig.m,TriTConfig.DetectorL,TriTConfig.FS_Num,'CG');
    end
else
    if TriTConfig.DP == 1
        ConcPred = FunctionofTripleT(deltaV(1),deltaV(2),deltaV(3),1,1,1,TriTErrinput(4),TriTErrinput(5),TriTErrinput(6),TriTErrinput(7),TriTErrinput(8),TriTErrinput(9),TriTConfig.m,TriTConfig.DetectorL,TriTConfig.FS_Num,TriTConfig.CG_Type,TriTConfig.ratio);
    else
        ConcPred = FunctionofTripleT(deltaV(1),deltaV(2),deltaV(3),1,1,1,TriTErrinput(4),TriTErrinput(5),TriTErrinput(6),TriTErrinput(7),TriTErrinput(8),TriTErrinput(9),TriTConfig.m,TriTConfig.DetectorL,TriTConfig.FS_Num,TriTConfig.CG_Type);
    end
end