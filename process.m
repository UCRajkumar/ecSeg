clear;
%Code to quantify ecDNA detection using precision and recall
%The predictions undergo post-processing (cleaning stray pixels, spurs,
%etc). Small ecDNA are slightly enlarged for better visual detection. The
%centers of test_mask ecDNA and the prediction ecDNA are compared with a
%tolerance of 5 pixel distances. 

img = dir('./predictions/*.npy');
precision = [];
recall = [];
tp_f = [];
fp_f = [];
fn_f = [];
for i = 1:length(img)
    ec_val = [];
    tp = 0;
    fp = 0;
    fn = 0;
    [~, a, ~] = fileparts(img(i).name);
    [~, b, ~] = fileparts(a);
    masker = imread(['./test_mask/' b '_mask.tif']);
    mask_i = masker(:,:,4);
    mask = bwmorph(mask_i, 'shrink', Inf);
    [mask_y, mask_x] = find(mask);
    pred = method2(readNPY(['./predictions/' img(i).name]));
    pred = bwmorph(clean(bwareafilt(pred == 1, [15 150])), 'shrink', Inf);
    [pred_y, pred_x] = find(pred);
    for j = 1:length(mask_x)
        for k = 1:length(pred_x)
            if(pred_x(k) ~= 0)
                X = [mask_x(j), mask_y(j); pred_x(k), pred_y(k)];
                if(pdist(X, 'euclidean') <= 5)
                    tp = tp+1;
                    pred_x(k) = 0;
                    break;
                end
            end
        end
    end
    fp = length(pred_x) - tp;
    fn = length(mask_x) - tp;
    if(fp < 0 | fn < 0)
        disp(b);
        break;
    end  
    tp_f = [tp_f tp];
    fp_f = [fp_f fp];
    fn_f = [fn_f fn];
end

[precision, recall] = stat(sum(tp_f), sum(fp_f), sum(fn_f))

function [precision, recall] = stat(TP, FP, FN)
    precision = TP /(TP + FP);
    recall = TP /  (TP + FN);
end

function [m2] = method2(I)
    m2 = zeros(length(I),length(I));
    for i = 1:length(I)
        for j = 1:length(I)
            if(I(i,j,3) > I(i,j,4) && I(i,j,3) > I(i,j,2) && I(i,j,3) > I(i,j,1))
                m2(i,j) = 2;
            end
            if(I(i,j,2) > I(i,j,4) && I(i,j,2) > I(i,j,3) && I(i,j,2) > I(i,j,1))
                m2(i,j) = 3;
            end
            if(I(i,j,4) > I(i,j,3) && I(i,j,4) > I(i,j,2) && I(i,j,4) > I(i,j,1))
                m2(i,j) = 1;
            end
        end
    end
    temp = imdilate(imfill((m2 == 3), 'holes'), strel('diamond', 1));
    m2(temp == 1) = 3;
    temp = imdilate(imfill((m2 == 2), 'holes'), strel('diamond', 1));
    m2(temp == 1) = 2;
end

function [bw] = clean(bw)
    bw = imfill(bw,'holes');
    bw = bwmorph(bw,'clean');
    bw = bwmorph(bw,'fill');
    bw = bwmorph(bw, 'open');
    bw = bwmorph(bw, 'spur');
end