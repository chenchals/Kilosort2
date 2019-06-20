function bytes = get_file_size(fname, headerBytes)
% gets file size ensuring that symlinks are dereferenced
% Modifications:
%     Get nBytes from dir call and subtract headerBytes
    d = dir(fname);
    bytes = d(1).bytes - headerBytes;
    % in case more than 1 file (non .bin files like .sev files)
    bytes = bytes*size(d,1);
end
