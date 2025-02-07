% Program: DD Data Processing
% Author: Hy Truong Son
% Position: PhD student
% Institution: Department of Computer Science, The University of Chicago
% Email: hytruongson@uchicago.edu, sonpascal93@gmail.com
% Website: http://people.inf.elte.hu/hytruongson/
% Copyright 2016 (c) Hy Truong Son. All rights reserved. Only for academic
% purposes.

% Data structure
% --------------
% First line: Number of samples
% For each sample:
% - Number of vertices N
% - The next line contains N numbers that is label of the vertex
% - The next N lines, each line contains a number of adjacent vertices,
% then a list of (vertex, weight). By default, weight = 1.
% - The last line is the type of the enzyme

function [] = dd_data_processing()
    load('DD.mat', 'DD', 'ldd');
    nSamples = size(DD, 2);
    fid = fopen('DD.dat', 'w');
    fprintf(fid, '%d\n', nSamples);
    for sample = 1 : nSamples
        N = size(DD(sample).al, 1);
        fprintf(fid, '%d\n', N);
        for i = 1 : N
            label = DD(sample).nl.values(i);
            fprintf(fid, '%d ', label);
        end
        fprintf(fid, '\n');
        for i = 1 : N
            nAdj = size(DD(sample).al{i, 1}, 2);
            fprintf(fid, '%d ', nAdj);
            for j = 1 : nAdj
                vertex = DD(sample).al{i, 1}(j);
                weight = 1;
                fprintf(fid, '%d %d ', vertex, weight);
            end
            fprintf(fid, '\n');
        end
        type = ldd(sample);
        fprintf(fid, '%d\n', type);
    end
    fclose(fid);
end
