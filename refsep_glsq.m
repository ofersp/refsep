function [X_hat,Y_hat] = refsep_glsq(X0,Z,max_iters,gmm,given_comps,show_progress)

if ~exist('show_progress','var'); show_progress = true; end
im_sz = size(Z);
X_hat = X0(:);
Y_hat = Z(:)-X_hat;

if show_progress
  hfig = figure(1);
  hwb = waitbar_create();
end

% perform some global least squares iterations
for i=1:max_iters
  if show_progress
    if show_preview(hfig,hwb,X_hat,Y_hat,im_sz,i,max_iters); 
      break; end; 
  end
  [X_hat,Y_hat] = map_gmm_glsq(Z,gmm,given_comps,X_hat,Y_hat);
end

if show_progress; waitbars_delete_all(); end
X_hat = reshape(X_hat,im_sz);
Y_hat = Z - X_hat;


function should_break = show_preview(hfig,hwb,X_hat,Y_hat,im_sz,i,max_iters)

set(0,'CurrentFigure',hfig)
subplot(1,2,1); imagesc(reshape(X_hat,im_sz(1),im_sz(2))); 
axis image; colormap gray; axis off;
subplot(1,2,2); imagesc(reshape(Y_hat,im_sz(1),im_sz(2))); 
axis image; colormap gray; axis off;
should_break = getappdata(hwb,'canceling');
waitbar(i/max_iters);
drawnow;


function hwb = waitbar_create()

hwb = waitbar(0,'GLSQ iteration progress','CreateCancelBtn',...
'setappdata(gcbf,''canceling'',1)');


function waitbars_delete_all()

F = findall(0,'type','figure','tag','TMWWaitbar'); 
delete(F);