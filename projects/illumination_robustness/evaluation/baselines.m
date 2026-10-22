function run_id = baselines(mode, benchmark_root, output_dir)
% Run segmentation-only MATLAB-compatible Otsu/Sobel baselines.
% This is a terminal-callable reference implementation. Classification (mu3)
% is intentionally excluded; outputs contain Q_seg = mu1*mu2.

if nargin < 1, mode = 'full'; end
if nargin < 2, benchmark_root = fullfile(pwd, 'hardware_items'); end
if nargin < 3 || isempty(output_dir)
    run_id = ['baselines_matlab_' mode '_' datestr(now, 'yyyymmdd_HHMMSS')];
    output_dir = fullfile(pwd, 'results', run_id);
else
    [~, run_id] = fileparts(output_dir);
end
if ~exist(output_dir, 'dir'), mkdir(output_dir); end
metrics_dir = fullfile(output_dir, 'metrics');
if ~exist(metrics_dir, 'dir'), mkdir(metrics_dir); end

csv_path = fullfile(metrics_dir, 'per_image.csv');

scenes = 1:4; shadings = 1:13; noises = 1:14;
if strcmpi(mode, 'smoke')
    tuples = [1 1 1; 1 7 7; 1 13 14];
elseif strcmpi(mode, 'series')
    tuples = zeros(numel(scenes) * 13, 3); k = 1;
    for r=scenes
        for b=1:13
            tuples(k,:) = [r b b+1]; k = k + 1;
        end
    end
else
    tuples = zeros(numel(scenes)*numel(shadings)*numel(noises), 3); k=1;
    for r=scenes, for b=shadings, for n=noises
        tuples(k,:)=[r b n]; k=k+1;
    end, end, end
end

scene_gt = cell(1, 4);
for r=1:4
    gt_data = load(fullfile(benchmark_root, 'truth', sprintf('benchdata_%d_gd.mat', r)));
    scene_gt{r} = gt_data.ground_truth;
end
n_images = size(tuples, 1);
rows = cell(n_images, 2);
start_time = tic; progress_count = 0;
use_parallel = exist('parpool', 'file') == 2 && license('test', 'Distrib_Computing_Toolbox');
if use_parallel
    pool = gcp('nocreate');
    if isempty(pool), pool = parpool('local'); end
    queue = parallel.pool.DataQueue;
    afterEach(queue, @progress_update);
    fprintf('Using %d parallel workers.\n', pool.NumWorkers);
else
    fprintf('Parallel Computing Toolbox unavailable; using serial execution.\n');
end
if use_parallel
    parfor k=1:n_images
        rows(k,:) = evaluate_tuple(tuples(k,:), benchmark_root, scene_gt);
        send(queue, 1);
    end
else
    for k=1:n_images
        rows(k,:) = evaluate_tuple(tuples(k,:), benchmark_root, scene_gt);
        progress_update(1);
    end
end
fid = fopen(csv_path, 'w');
fprintf(fid, 'method,scene,shading,noise,image_id,dice,iou,q_seg,mu1,mu2,q1,q2,gt_object_count,predicted_object_count\n');
for k=1:n_images
    fprintf(fid, '%s\n%s\n', rows{k,1}, rows{k,2});
end
fclose(fid);
fprintf('Saved MATLAB-compatible baseline results to %s\n', csv_path);

    function progress_update(~)
        progress_count = progress_count + 1;
        if progress_count == 1 || mod(progress_count, 10) == 0 || progress_count == n_images
            elapsed = toc(start_time); rate = progress_count / max(elapsed, eps);
            remaining = (n_images - progress_count) / max(rate, eps);
            fprintf('Progress: %d/%d (%.1f%%), %.1f images/s, ETA %s\n', ...
                progress_count, n_images, 100*progress_count/n_images, rate, format_duration(remaining));
        end
    end
end

function output = evaluate_tuple(tuple, benchmark_root, scene_gt)
r=tuple(1); b=tuple(2); n=tuple(3);
image = imread(fullfile(benchmark_root, 'data', sprintf('benchdata_%02d_%02d_%02d.tif', r,b,n)));
gt = resize_ground_truth(logical(scene_gt{r}), size(image));
[p_otsu, s_otsu] = otsu_reference(image, gt);
[p_sobel, s_sobel] = sobel_reference(image, gt);
image_id = sprintf('r%02d_b%02d_n%02d', r,b,n);
output = {make_row('otsu',r,b,n,image_id,p_otsu,gt,s_otsu), make_row('sobel',r,b,n,image_id,p_sobel,gt,s_sobel)};
end

function row = make_row(method,r,b,n,id,pred,gt,s)
row = sprintf('%s,%d,%d,%d,%s,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%d,%d', ...
    method,r,b,n,id,s.dice,s.iou,s.q_seg,s.mu1,s.mu2,s.q1,s.q2,s.gt_object_count,s.predicted_object_count);
end

function text = format_duration(seconds)
if seconds < 60, text=sprintf('%.0fs',seconds); return; end
minutes=floor(seconds/60); seconds=round(seconds-60*minutes);
if minutes < 60, text=sprintf('%dm %02ds',minutes,seconds); return; end
hours=floor(minutes/60); minutes=minutes-60*hours;
text=sprintf('%dh %02dm',hours,minutes);
end

function gt = resize_ground_truth(gt, target_size)
if ~isequal(size(gt), target_size(1:2))
    gt = imresize(gt, target_size(1:2), 'bicubic') > 0;
end
end

function [mask, score] = otsu_reference(image, gt)
norm = normalize_reference(image);
thresholds = 0.01:0.04:0.99;
best = -inf; mask = false(size(norm)); score=[];
for t=thresholds
    candidate = postprocess_otsu(norm > t);
    current = segmentation_score(candidate, gt);
    if current.q_seg > best, best=current.q_seg; mask=candidate; score=current; end
end
end

function [mask, score] = sobel_reference(image, gt)
norm = normalize_reference(image);
[~, base_threshold] = edge(norm, 'sobel');
factors = 0:0.05:1.0;
best=-inf; mask=false(size(norm)); score=[];
for f=factors
    candidate = postprocess_sobel(edge(norm, 'sobel', base_threshold*f));
    current = segmentation_score(candidate, gt);
    if current.q_seg > best, best=current.q_seg; mask=candidate; score=current; end
end
end

function out = normalize_reference(image)
x = double(image); out = imcomplement(uint8(x)); out = im2double(out);
end

function out = postprocess_otsu(mask)
out = imfill(mask,'holes'); out=imdilate(out,ones(5)); out=imfill(out,'holes');
out=imopen(out,strel('disk',9)); out=imclearborder(imfill(out,'holes'));
out=bwareaopen(out,100,8);
end

function out = postprocess_sobel(mask)
out=imfill(mask,'holes'); out=imdilate(out,ones(3)); out=imfill(out,'holes');
out=imopen(out,strel('disk',5)); out=imclearborder(imfill(out,'holes'));
out=bwareaopen(out,100,8);
end

function s = segmentation_score(pred, gt)
pred=logical(pred); gt=logical(gt); Lp=bwlabel(pred,8); Lg=bwlabel(gt,8);
n_gt=max(Lg(:)); n_pred=max(Lp(:)); matched=0;
for u=1:n_pred
    blob=(Lp==u); overlap=zeros(1,n_gt);
    for v=1:n_gt, overlap(v)=nnz(blob & (Lg==v)); end
    if max(overlap) > nnz(blob)/3, matched=matched+1; end
end
q1=abs(n_gt-matched)/max(n_gt,1); union=nnz(pred|gt);
q2=nnz(xor(pred,gt))/max(union,1); mu1=zmf_quality(q1); mu2=zmf_quality(q2);
s.q1=q1; s.q2=q2; s.mu1=mu1; s.mu2=mu2; s.q_seg=mu1*mu2;
s.dice=2*nnz(pred & gt)/(nnz(pred)+nnz(gt)); s.iou=nnz(pred & gt)/max(union,1);
s.gt_object_count=n_gt; s.predicted_object_count=n_pred;
end

function y=zmf_quality(x)
if x<=0, y=1; elseif x>=1, y=0; elseif x<=0.5, y=1-2*x^2; else, y=2*(1-x)^2; end
end

function write_row(fid,method,r,b,n,id,pred,gt,s)
fprintf(fid,'%s,%d,%d,%d,%s,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%d,%d\n',method,r,b,n,id,s.dice,s.iou,s.q_seg,s.mu1,s.mu2,s.q1,s.q2,s.gt_object_count,s.predicted_object_count);
end
