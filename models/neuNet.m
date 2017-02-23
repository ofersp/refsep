function [net] = neuNet(layers_spec, input_shape)
% create a neural network according to the given layers spec and input
% shape.
% input_shape = [rows,columns,channels,groups]
% The difference between 'channels' and 'groups' is that convolutions are 
% applied to all the channels together and to all the groups separately.
% The shape of inputs in all layer is always 5d:
% [rows,columns,channels,groups,samples]
% samples is set through the setMBSize function which is called every time a new sample size is given to forward().
%
% net.forward() - compute a forward pass over the network (input is in 5d shape)
% net.backward() - propagates the gradient backwards.
%
% Dan Rosenbaum http://www.cs.huji.ac.il/~danrsm
%


net.spec = layers_spec;
net.input_shape = input_shape;
net.layers = {};
net.theta={};
for li = 1:length(layers_spec)
    [net,input_shape] = add_layer(net,layers_spec{li},input_shape);
end
net.setMBSize = @(net,N) setMBSize(net,N);
net.forward = @(net,X) forward(net,dataType(X));
net.backward = @(net,lin,delta) backward(net,lin,dataType(delta));
net.MBsize=1;
net.out_scale = 1;

function net = setMBSize(net,N)
for li=1:length(net.layers)
   if (strcmp(net.layers{li}.type,'reshape'))
       n5 = reshape((0:N-1)*prod(net.layers{li}.in_shape),[1,1,1,1,N]);
       idx = bsxfun(@plus,repmat(net.layers{li}.idx,[1,1,1,1,N]),n5);
       net.layers{li}.forward = @(in,theta) Reshape_forward(in,idx);
       [sidx,inv_idx,sum_idx] = calcInvMap(idx);
       net.layers{li}.backward = @(err_out,in,theta) ...
           Reshape_backward(err_out,sidx,inv_idx,sum_idx,net.layers{li}.in_shape);
        
   end
end
net.MBsize=N;

%%
% given an input for the network, perform a forward pass and returns
% output and intermediate calculations.
function [o,lin] = forward(net, X)

if (nargout==2), lin = cell(size(net.layers)); end
if (net.MBsize~=size(X,5)), net=net.setMBSize(net,size(X,5)); end;
o = X;
for l=1:length(net.layers)
    if exist('lin','var'), lin{l}=o; end
    o = net.layers{l}.forward(o,net.theta{l});
end

%%
% calculate gradient of weights
function dtheta = backward(net, lin, delta)
dtheta = cell(size(net.layers));
for l=length(net.layers):-1:1
    [delta,dtheta{l}] = net.layers{l}.backward(delta,lin{l},net.theta{l});
end

%%
% add layers according to spec
function [net, out_shape] = add_layer(net, lspec, in_shape)

assert(length(in_shape)==4);

layer=struct();
theta=struct();
layer.type = lspec.type;
layer.in_shape = in_shape;
switch lower(lspec.type)
    case 'affine'
        assert(length(lspec.out_shape)==1);
        [theta.W, theta.b] = rand_init(lspec.out_shape,prod(in_shape(1:3)));
        layer.forward = @(in,theta) Affine_forward(in,theta,in_shape);
        layer.backward = @(err_out,in,theta) Affine_backward(err_out,in,theta,in_shape);
        out_shape = [lspec.out_shape,1,1,in_shape(4)];        
    case 'relu'
        layer.forward = @(in,theta) Relu_forward(in);
        layer.backward = @(err_out,in,theta) Relu_backward(err_out, in);
        out_shape = in_shape;     
    case 'square'
        layer.forward = @(in,theta) Square_forward(in);
        layer.backward = @(err_out,in,theta) Square_backward(err_out, in);
        out_shape = in_shape;     
    case 'reshape'
        [sidx,inv_idx,sum_idx] = calcInvMap(lspec.idx);
        layer.idx = lspec.idx;
        layer.forward = @(in,theta) Reshape_forward(in,lspec.idx);
        layer.backward = @(err_out,in,theta) Reshape_backward(err_out,sidx,inv_idx,sum_idx,in_shape);
        out_shape = [size(lspec.idx,1),size(lspec.idx,2),size(lspec.idx,3),size(lspec.idx,4)];
    case 'max1'
        layer.forward = @(in,theta) Max1_forward(in);
        layer.backward = @(err_out,in,theta) Max1_backward(err_out, in);
        out_shape = [1,in_shape(2:4)];
    case 'conv'
        [i2c_idx,c2i_idx] = calcIdx(in_shape, lspec.kernel_size, lspec.stride,1);
        i2c_idx = reshape(i2c_idx,[size(i2c_idx,1),1,1,size(i2c_idx,2)]);
        [net,out_shape] = add_layer(net,struct('type','reshape','idx',i2c_idx),in_shape);
        [net,out_shape] = add_layer(net,struct('type','affine','out_shape',lspec.kernel_size(3)),out_shape);
        [net,out_shape] = add_layer(net,struct('type','reshape','idx',c2i_idx),out_shape);
        return;
    case 'maxpool'
        [i2c_idx,c2i_idx] = calcIdx(in_shape,lspec.kernel_size,lspec.stride,0);
        [net,out_shape] = add_layer(net, struct('type','reshape','idx',i2c_idx), in_shape);
        [net,out_shape] = add_layer(net,struct('type','max1'), out_shape);
        [net,out_shape] = add_layer(net,struct('type','reshape','idx',c2i_idx),out_shape);
        return;
end

net.layers{end+1}=layer;
net.theta{end+1}=theta;


%%
% Affine
function out = Affine_forward(in, theta,in_shape)
inMat = reshape(in,[prod(in_shape(1:3)),numel(in)/prod(in_shape(1:3))]);
out = bsxfun(@plus,theta.W*inMat,theta.b);
out = reshape(out,[size(out,1),1,1,in_shape(4),size(in,5)]);

function [err_in, dtheta] = Affine_backward(err_out, in, theta, in_shape)
in = reshape(in,[prod(in_shape(1:3)),numel(in)/prod(in_shape(1:3))]);
err_out = reshape(err_out,[size(theta.W,1),numel(err_out)/size(theta.W,1)]);
err_in = theta.W'*err_out;
err_in = reshape(err_in,[in_shape,numel(err_in)/prod(in_shape)]);
dtheta=struct();
dtheta.W = err_out*in';
dtheta.b = sum(err_out,2);

%%
% Relu
function out = Relu_forward(in)
out = max(in,0);

function [err_in, dtheta] = Relu_backward(err_out, in)
err_in = (in>0).*err_out;
dtheta=struct();

%%
% Square
function out = Square_forward(in)
out = in.^2;

function [err_in, dtheta] = Square_backward(err_out, in)
err_in = 2*in.*err_out;
dtheta=struct();

%%
% Reshape
function out = Reshape_forward(in,idx)
out=in(idx);

function [err_in, dtheta] = Reshape_backward(err_out, sidx, inv_idx, sum_idx,in_shape)
err_in = dataType(zeros([in_shape,size(err_out,5)]));
cs=cumsum(err_out(inv_idx));
sumvec=cs(sum_idx)'-[0,cs(sum_idx(1:end-1))'];
dtheta=struct();

%% 
% Max1
function out = Max1_forward(in)
out = max(in,[],1);

function [err_in, dtheta] = Max1_backward(err_out, in)
err_in = bsxfun(@times,bsxfun(@eq,in,max(in,[],1)),err_out);
dtheta=struct();

%% 
% weight initialization
function [W,b] = rand_init(r,c)
R=randn(r,c);
if (r<c), W = dataType(orth(R')');
else W = dataType(orth(R)); end
b = dataType(zeros(r,1));

%%
% manage data type (precision, GPU/CPU)
function B = dataType(A)
B = single(A);

% calculate the idx for reshaping the data to fit convolutions and maxpool 
% (like im2col and col2im)
function [i2cidx,c2iidx] = calcIdx(imsize, kersize, stride,flattenChannels)

% single channel im2col idx
i2cidx1 = im2col(reshape(1:prod(imsize([1,2])),imsize([1,2])),kersize([1,2]));

% modify according to stride
strideIdx = zeros(imsize(1)-kersize(1)+1,imsize(2)-kersize(2)+1);
strideIdx(1:stride:end,1:stride:end)=1;
i2cidx1=i2cidx1(:,strideIdx(:)==1);
    
if (flattenChannels) % used for convolution layers
 
    % multiply for all channels
    channel_idx = reshape((0:imsize(3)-1)*prod(imsize([1,2])),[1,1,imsize(3)]);
    i2cidx2 = bsxfun(@plus,repmat(i2cidx1,[1,1,imsize(3)]),channel_idx);
    i2cidx2 = permute(i2cidx2,[1,3,2]);
    i2cidx2 = reshape(i2cidx2,[size(i2cidx2,1)*size(i2cidx2,2),size(i2cidx2,3)]);
    
    % multiply for all groups
    group_idx = reshape((0:imsize(4)-1)*prod(imsize([1,2,3])),[1,1,imsize(4)]);
    i2cidx = bsxfun(@plus,repmat(i2cidx2,[1,1,imsize(4)]),group_idx);
    i2cidx = reshape(i2cidx,[size(i2cidx,1),size(i2cidx,2)*size(i2cidx,3)]);
        
    % calc col2im idx
    WIpidx = reshape(1:(kersize(3)*size(i2cidx,2)),[kersize(3),size(i2cidx,2)]);
    c2iidx = reshape(WIpidx,[kersize(3),sum(strideIdx(:,1)),sum(strideIdx(1,:)),imsize(4)]);
    c2iidx = permute(c2iidx,[2,3,1,4]);

else % used for max-pool layers

    % multiply for all channels and groups
    ncg = imsize(3)*imsize(4);
    channel_idx = reshape((0:ncg-1)*prod(imsize([1,2])),[1,1,ncg]);
    i2cidx = bsxfun(@plus,repmat(i2cidx1,[1,1,ncg]),channel_idx);
    i2cidx = reshape(i2cidx,[size(i2cidx,1),size(i2cidx,2)*size(i2cidx,3)]);

    % calc col2im idx
    c2iidx = reshape(1:size(i2cidx,2),[sum(strideIdx(:,1)),sum(strideIdx(1,:)),imsize(3),imsize(4)]);
end

%%
% Calculate idx for inverse mapping. Based on code by Shai Shalev-Shwartz
function [sidx,inv_idx,sum_idx]=calcInvMap(idx)
[sidx,inv_idx] = sort(idx(:));
sum_idx = [find(sidx(2:end)~=sidx(1:end-1))',length(sidx)];
sidx = unique(sidx);
