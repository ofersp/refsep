addpath models;

Z = im2double(imread('examples/window_disp.png'));

params.use_gating = true;
params.show_progress = true;
params.show_results = true;
params.load_annot_file = 'examples/window_disp_annot.mat';

[X,Y] = refsep(Z,[],[],params);