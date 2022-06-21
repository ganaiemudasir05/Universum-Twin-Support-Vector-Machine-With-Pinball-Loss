function [Predict_Y, time1] = Pin_UTWSVM(TestX, DataTrain, U, FunPara)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pin_UTWSVM: Universum Twin Support Vector Machine With Pinball Loss
%
% Predict_Y = pinGTSVM(TestX,DataTrain,FunPara)
%
% Input:
%    TestX - Testing Data Matrix, with labels appended in last column. Each row vector of fea is a data point. 
%    DataTrain- Training Data Matrix, with labels appended in last column 
%    U- Universum Data Matrix.
%    FunPara- struct containing the hyperparameters
% Reference:
%      Universum twin support vector machine with pinball loss function
%
%
%  Written by: Mudasir
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % L=load('fertility.txt');
% % %K=L(randperm(100),:);
% % DataTrain=L(1:70,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initailization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic;
[no_input, no_col] = size(DataTrain);
obs = DataTrain(:, no_col);
A = DataTrain(obs == 1, 1:end - 1);
B = DataTrain(obs ~= 1, 1:end - 1);

% % FunPara=struct('c1',0.5,'c2',0.5,'kerfPara',struct('type','rbf','pars',2^2));
c1 = FunPara.c1;
c3 = FunPara.c3;

c2 = FunPara.c2;
c4 = FunPara.c4;

t1 = FunPara.t1;
t2 = FunPara.t2;

t3 = FunPara.t3;
t4 = FunPara.t4;

epsilon = FunPara.e;

kerfPara = FunPara.kerfPara;
eps1 = 10^ - 4;
eps2 = 10^ - 4;
% eps1 = c1;
% eps2 = c1;

m1 = size(A, 1);
m2 = size(B, 1);
m3 = size(U, 1);
m = m1 + m2;
um = size(U, 1);
e1 = ones(m1, 1);
e2 = ones(m2, 1);
eu = ones(um, 1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute Kernel
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(kerfPara.type, 'lin')
    X1 = [A, e1];
    X2 = [B, e2];
    X3 = [U, eu];
else
    X = [A; B];
    %         X1 = [kernelfun(A, kerfPara, X), e1];
    %         X2 = [kernelfun(B, kerfPara, X), e2];
    %         X3 = [kernelfun(U, kerfPara, X), e2];
    
    mu = kerfPara.pars(1);
    K = zeros(m1, m);
    
    for i = 1:m1
        
        for j = 1:m
            nom = norm(A(i, :) - X(j, :));
            K(i, j) = exp(-1 / (2 * mu * mu) * nom * nom);
        end
        
    end
    
    X1 = [K e1];
    
    K = zeros(m2, m);
    
    for i = 1:m2
        
        for j = 1:m
            nom = norm(B(i, :) - X(j, :));
            K(i, j) = exp(-1 / (2 * mu * mu) * nom * nom);
        end
        
    end
    
    X2 = [K e2];
    
    K = zeros(m3, m);
    
    for i = 1:m3
        
        for j = 1:m
            nom = norm(U(i, :) - X(j, :));
            K(i, j) = exp(-1 / (2 * mu * mu) * nom * nom);
        end
        
    end
    
    X3 = [K eu];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute (w1,b1) and (w2,b2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%DpinGTSVM1
HH = X1' * X1;
G = [X2; -X3];
HH = HH + eps1 * eye(size(HH)); %regularization
HHG = HH \ G';
kerH1 = G * HHG;
kerH1 = (kerH1 + kerH1') / 2;
f1 = [e2; (-1 + epsilon) * eu]';
lb = [-t1 * c1 * e2; -t2 * c2 * eu];
ub = [c1 * e2; c2 * eu];
alpha1 = quadprog(kerH1, -f1, [], [], [], [], lb, ub, [], optimset('Display', 'off')); %SOR
vpos = -HHG * alpha1;

%%%%DpinGTSVM2
QQ = X2' * X2;
QQ = QQ + eps2 * eye(size(QQ)); %regularization
H = [X1; -X3];
QQP = QQ \ H';
kerH1 = H * QQP;
kerH1 = (kerH1 + kerH1') / 2;
f2 = [e1; (-1 + epsilon) * eu]';
lb2 = [-t3 * c3 * e1; -t4 * c4 * eu];
ub2 = [c3 * e2; c4 * eu];
gamma1 = quadprog(kerH1, -f2, [], [], [], [], lb2, ub2, [], optimset('Display', 'off'));
vneg = QQP * gamma1;
% % clear kerH1 H G HH HHG QQ QQP;
w1 = vpos(1:(length(vpos) - 1));
b1 = vpos(length(vpos));
w2 = vneg(1:(length(vneg) - 1));
b2 = vneg(length(vneg));
time1 = toc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Predict and output
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TestX_labels=TestX(:,end);
TestX=TestX(:,1:end-1);
%m=size(TestX,1);
if strcmp(kerfPara.type,'lin')
    m=size(TestX,1);
    H=TestX;
    w11=sqrt(w1'*w1);
    w22=sqrt(w2'*w2);
    y1=H*w1+b1*ones(m,1);
    y2=H*w2+b2*ones(m,1);
else
    C=[A;B];
    % H=kernelfun(TestX,kerfPara,C);
    %%%%%%%%%%%%%%%%%%%%%%%5
    mu = kerfPara.pars(1);
    m1 = size(TestX,1);
    K = zeros(m1, m);
    for i = 1:m1
        
        for j = 1:m
            nom = norm(TestX(i, :) - C(j, :));
            K(i, j) = exp(-1 / (2 * mu * mu) * nom * nom);
        end
        
    end
    
    H = K;
    
    K = zeros(m, m);
    for i = 1:m
        
        for j = 1:m
            nom = norm(C(i, :) - C(j, :));
            K(i, j) = exp(-1 / (2 * mu * mu) * nom * nom);
        end
        
    end
    TempPPP = K;
    %%%%%%%%%%%%%%%%%%%%%%%5
    
    w11=sqrt(w1'*TempPPP*w1);
    w22=sqrt(w2'*TempPPP*w2);
    y1=H*w1+b1*ones(m1,1);
    y2=H*w2+b2*ones(m1,1);
end
clear H; clear C;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m1=y1/w11;
m2=y2/w22;
Predict_Y = sign(abs(m2)-abs(m1));
end
