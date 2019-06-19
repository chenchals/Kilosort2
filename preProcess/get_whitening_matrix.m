function Wrot = get_whitening_matrix(rez)
% Modifications:
%    for DataAdapter
%    logic:
%    Replaced
%        offset = max(0, twind + 2.*NchanTOT*((NT - ops.ntbuff) *
%          (ibatch-1) - 2*ops.ntbuff)); with
%        offset = max(0, twind + ops.dataTypeBytes*NchanTOT*((NT -
%          ops.ntbuff) * (ibatch-1) - ops.dataTypeBytes*ops.ntbuff)); 
%

ops = rez.ops;
Nbatch = ops.Nbatch;
twind = ops.twind;
NchanTOT = ops.NchanTOT;
NT = ops.NT;
NTbuff = ops.NTbuff;
chanMap = ops.chanMap;
Nchan = rez.ops.Nchan;
xc = rez.xc;
yc = rez.yc;

% load data into patches, filter, compute covariance
if isfield(ops,'fslow')&&ops.fslow<ops.fs/2
    [b1, a1] = butter(3, [ops.fshigh/ops.fs,ops.fslow/ops.fs]*2, 'bandpass');
else
    [b1, a1] = butter(3, ops.fshigh/ops.fs*2, 'high');
end

fprintf('Getting channel whitening matrix... \n');
if ops.GPU
    CC = gpuArray.zeros( Nchan,  Nchan, 'single');
else
    CC = zeros( Nchan,  Nchan, 'single');
end

ibatch = 1;
while ibatch<=Nbatch    
    %offset = max(0, twind + 2.*NchanTOT*((NT - ops.ntbuff) * (ibatch-1) - 2*ops.ntbuff));
    offset = max(0,...
        twind ...
        + ops.dataTypeBytes*NchanTOT*((NT - ops.ntbuff) * (ibatch-1)...
        - ops.dataTypeBytes*ops.ntbuff)...
        );

     buff = ops.dataAdapter.batchRead(offset,ops.NchanTOT, NTbuff, ops.dataTypeString);
    
    if isempty(buff)
        break;
    end
    nsampcurr = size(buff,2);
    if nsampcurr<NTbuff
        buff(:, nsampcurr+1:NTbuff) = repmat(buff(:,nsampcurr), 1, NTbuff-nsampcurr);
    end
    if ops.GPU
        dataRAW = gpuArray(buff);
    else
        dataRAW = buff;
    end
    dataRAW = dataRAW';
    dataRAW = single(dataRAW);
    dataRAW = dataRAW(:, chanMap);
    
    % subtract the mean from each channel
    dataRAW = dataRAW - mean(dataRAW, 1);

    datr = filter(b1, a1, dataRAW);
    datr = flipud(datr);
    datr = filter(b1, a1, datr);
    datr = flipud(datr);

    % CAR, common average referencing by median
    if getOr(ops, 'CAR', 1)
        datr = datr - median(datr, 2);
    end
    
    CC        = CC + (datr' * datr)/NT;    
    
    ibatch = ibatch + ops.nSkipCov;
end
CC = CC / ceil((Nbatch-1)/ops.nSkipCov);

fprintf('Channel-whitening filters computed. \n');

if ops.whiteningRange<Inf
    ops.whiteningRange = min(ops.whiteningRange, Nchan);
    Wrot = whiteningLocal(gather_try(CC), yc, xc, ops.whiteningRange);
else
    [E, D] 	= svd(CC);
    D       = diag(D);
%     eps 	= mean(D); %1e-6;
    eps 	= 1e-6;
    f  = mean((D+eps) ./ (D+1e-6));
%     fprintf('%2.2f ', f)
    Wrot 	= E * diag(f./(D + eps).^.5) * E';
end
Wrot    = ops.scaleproc * Wrot;
