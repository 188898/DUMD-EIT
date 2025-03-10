
fmdl= ng_mk_ellip_models([0,6,6,0.1],[0;45;90;135;180;225;270;315],[0.2,0,0.03]); 
background = 1.0;
[stim,msel] = mk_stim_patterns(8,1,[0,1],[0,1],{},0.01);
imdl = mk_common_model('f2c2',8);
imdl.fwd_model = fmdl;
imdl.fwd_model.stimulation = stim;
imdl.fwd_model.meas_select = msel;
imdl.hyperparameter.value = 0.00001;
img= mk_image(imdl.fwd_model, background);
vh= fwd_solve( img );vh = vh.meas; 

abnormal_r1 = rand() * 0.5 + 1;
while(1)
    tar1_X = 12 * rand() - 6;
    tar1_Y = 12 * rand() - 6;
    dis1 = distance([tar1_X,tar1_Y],[0,0]);
    if dis1 < (6 - abnormal_r1-0.2)
           break
    end
end
% Randomly set the conductivity of the abnormal region, range: 3 ~ 4 (the background conductivity is 1)
tar1_data = rand() + 3;

% Creating exception regions
target = mk_c2f_circ_mapping(img.fwd_model, [tar1_X;tar1_Y;abnormal_r1]);


% The sparse matrix is transformed into a regular matrix
A1 = full(target);

img.elem_data(:,:) = background;
% The anomalous region is set in the finite element model
for i = 1 : size(A1)
    if A1(i,:) ~= 0
        A1(i,:) = tar1_data;
        img.elem_data(i,:) = A1(i,:);    
    end
end

vi = fwd_solve(img); vi = vi.meas;   
img2= inv_solve(imdl, vh, vi);
subplot(121); 
show_fem(img);
subplot(122);
show_fem(img2);



