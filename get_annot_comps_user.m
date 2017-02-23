function given_comps = get_annot_comps_user(Z,gmm)

im_sz = size(Z);
patch_sz = [8,8];
given_comps_im.X = nan(im_sz);
given_comps_im.Y = nan(im_sz);

while true
  
  show_rects(Z,given_comps_im);  
  [ginp_x,ginp_y,btn] = ginput(1);
  xy = round([ginp_x,ginp_y]); 
  if btn == 's'
    [fname,pname] = uiputfile();
    if ~fname; continue; end
    save(fullfile(pname,fname),'given_comps_im');
  elseif btn == 'l'
    [fname,pname] = uigetfile();
    if ~fname; continue; end
    tmp = load(fullfile(pname,fname)); 
    given_comps_im = tmp.given_comps_im; clear tmp;
  elseif btn == 3
    break;
  elseif btn == 2
    L = ~isnan(given_comps_im.X) | ~isnan(given_comps_im.Y);
    [y,x] = ind2sub(im_sz,find(L));
    dists = sqrt(sum(bsxfun(@minus,[x,y],xy).^2,2));
    [min_dist,min_dist_ind] = min(dists);
    if min_dist > 8; continue; end
    given_comps_im.X(y(min_dist_ind),x(min_dist_ind)) = nan;
    given_comps_im.Y(y(min_dist_ind),x(min_dist_ind)) = nan;
  elseif btn == 1
    rectangle('Position',[xy(1)-3.5,xy(2)-3.5,8,8],'EdgeColor','b','LineWidth',0.5);
    patch = Z(xy(2)-3:xy(2)+4,xy(1)-3:xy(1)+4);
    patch = patch - mean(patch(:));
    axis off;
    drawnow;

    figure(3);
    gmm3 = gmm_condition_on_sum(gmm,patch(:));
    gmm3_means = bsxfun(@minus,gmm3.means,mean(gmm3.means));
    [~,sorted_comp_inds] = sort(gmm3.mixweights,'descend');
    means_X = gmm3_means(:,sorted_comp_inds(1:100));
    means_Y = bsxfun(@minus,patch(:),means_X);
    show_possible_decomps(means_X,means_Y,patch_sz,10,10);
    [ginp_x,ginp_y,btn] = ginput(1);
    if btn == 3; continue; end  
    comp_x = ceil(ginp_x/20); 
    comp_y = ceil(ginp_y/10);
    comp_ind3 = sorted_comp_inds(comp_y+10*(comp_x-1));
    [comp_ind1,comp_ind2] = ind2sub([gmm.nmodels,gmm.nmodels],comp_ind3);
    given_comps_im.X(xy(2),xy(1)) = comp_ind1;
    given_comps_im.Y(xy(2),xy(1)) = comp_ind2;
  end
end

given_comps = given_comps_im2col(given_comps_im, patch_sz);


function show_rects(Z,given_comps_im)

L = ~isnan(given_comps_im.X) | ~isnan(given_comps_im.Y);
im_sz = size(Z);
figure(2); 
imagesc(Z);
colormap gray; axis image; axis off;
[y,x] = ind2sub(im_sz,find(L));
for j=1:numel(y)
  rectangle('Position',[x(j)-3.5,y(j)-3.5,8,8],'EdgeColor','r','LineWidth',0.5);
end


function patch_im = show_possible_decomps(X,Y,patch_sz,num_rows,num_cols)

assert(size(X,2) == num_rows*num_cols);
cols_odd_even = [0.8 0.4 0.4; 0.6 0.4 0.4];
sz = patch_sz;
n = num_cols;
m = num_rows;

patch_im = zeros((sz(1)+2)*n,(sz(2)+2)*(2*m),3); % prepare background
for j=1:m
  for k=1:3
    patch_im(:,(1+(j-1)*2*(sz(2)+2)):(j*2*(sz(2)+2)),k) = cols_odd_even(mod(j,2)+1,k);
  end
end

for i=1:n % show decomp patch pairs
  for j=1:m*2
    if mod(j,2) == 0
      xy = reshape(X(:,i+(j/2-1)*n),sz) + 0.5;
    else
      xy = reshape(Y(:,i+((j+1)/2-1)*n),sz) + 0.5;
    end
    patch_im(2+(i-1)*(sz(1)+2):i*(sz(1)+2)-1,2+(j-1)*(sz(2)+2):j*(sz(2)+2)-1,:) = repmat(xy,1,1,3);
  end
end
imagesc(patch_im);
axis image; axis off;
