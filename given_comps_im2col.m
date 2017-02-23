function given_comps_col = given_comps_im2col(given_comps_im, patch_sz)

X_comps_col = im2col(given_comps_im.X,patch_sz,'sliding');
Y_comps_col = im2col(given_comps_im.Y,patch_sz,'sliding');
given_comps_col.X = X_comps_col(sub2ind(patch_sz,4,4),:);
given_comps_col.Y = Y_comps_col(sub2ind(patch_sz,4,4),:); 