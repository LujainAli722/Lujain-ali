%% ============================================================
% NeuroNova_ANN_Full.m
% Full ANN pipeline: data embedding, LOOCV hyperparameter search,
% final training, full metrics & plots, and prediction helper.
% ------------------------------------------------------------
clear; clc; close all;
fprintf('=== NeuroNova ANN | Full Code with All Calculations ===\n');


%% ================== 1) DATA EMBEDDING ==================
[T_feed, P_Feed, EthyleneFeed, EthaneFeed, PropaneFeed, Butene1Feed, ...
NumberTrays, RefluxRatio, DistillateRate, ColumnDiameter, ColumnSpacing, ...
T_Condenser, T_Reboiler, Duty_Condenser, Duty_Reboiler, Area_Cond, Area_Reb, ...
Top_Ethylene, Top_Ethane, Top_Propene, Top_1Butene, ...
Bot_Ethylene, Bot_Ethane, Bot_Propene, Bot_1Butene] = embedData();


[X, Y, inNames, outNames] = buildXY( ...
T_feed, P_Feed, EthyleneFeed, EthaneFeed, PropaneFeed, Butene1Feed, ...
NumberTrays, RefluxRatio, DistillateRate, ColumnDiameter, ColumnSpacing, ...
T_Condenser, T_Reboiler, Duty_Condenser, Duty_Reboiler, Area_Cond, Area_Reb, ...
Top_Ethylene, Top_Ethane, Top_Propene, Top_1Butene, ...
Bot_Ethylene, Bot_Ethane, Bot_Propene, Bot_1Butene);


nSamples = size(X,2);
fprintf('Samples: %d | Inputs: %d | Outputs: %d\n', nSamples, size(X,1), size(Y,1));
if nSamples <= 10
warning('Very small dataset. LOOCV will be used to estimate generalization.');
end


%% ================== 2) HYPERPARAMETERS GRID ==================
% عدّل الشبكات كما تريد
hiddenLayerGrid = {
[5], [10], [20], ...
[10 5], [10 10], [20 10], ...
[20 20]
};
maxEpochs = 1000;
goalMSE = 1e-6;
showWindow = false; % true لفتح نافذة التدريب لكل شبكة (بطيء مع البحث)


%% ================== 3) LOOCV SEARCH ==================
fprintf('\n--- Hyperparameter Search using LOOCV ---\n');
best = struct('hl', [], 'cvMSE', Inf, 'trainMSE', Inf, 'net', [], 'Yhat_cv', [], 'R2_cv', []);
for i = 1:numel(hiddenLayerGrid)
hl = hiddenLayerGrid{i};
[cvMSE, trainMSE, Yhat_cv, R2_cv] = loocv_score(X, Y, hl, maxEpochs, goalMSE, showWindow);
fprintf('Struct %-12s | CV-MSE = %.6g | Train-MSE = %.6g\n', mat2str(hl), cvMSE, trainMSE);
if cvMSE < best.cvMSE
best.hl = hl;
best.cvMSE = cvMSE;
best.trainMSE = trainMSE;
best.Yhat_cv = Yhat_cv;
best.R2_cv = R2_cv;
end
end


fprintf('\n>>> BEST STRUCT = %s | CV-MSE = %.6g\n', mat2str(best.hl), best.cvMSE);


%% ================== 4) FINAL TRAINING ON ALL DATA ==================
fprintf('\n--- Final Training on ALL data with best structure ---\n');
[finalNet, tr] = train_one(X, Y, best.hl, maxEpochs, goalMSE, true); % افتح نافذة التدريب النهائية


Yhat_final = finalNet(X);


% حساب المقاييس على كل البيانات:
metrics_final = compute_metrics(Y, Yhat_final);
fprintf('\n=== Final Model Metrics (All data) ===\n');
print_metrics(metrics_final, outNames);


%% ================== 5) PLOTS ==================
% Regression (كل المخرجات)
figure; plotregression(Y, Yhat_final); title('Regression (All outputs) - Final Model');


% R2 per output
figure;
bar(metrics_final.R2);
set(gca, 'XTick', 1:numel(outNames), 'XTickLabel', outNames, 'XTickLabelRotation', 45);
ylabel('R^2'); title('R^2 per Output'); grid on;


% Parity plots لكل مخرج
figure('Name','Parity plots');
nOut = size(Y,1);
for j = 1:nOut
subplot(ceil(nOut/2),2,j);
scatter(Y(j,:), Yhat_final(j,:), 60, 'filled'); hold on;
plot([min(Y(j,:)) max(Y(j,:))], [min(Y(j,:)) max(Y(j,:))], 'k--','LineWidth',1);
xlabel(sprintf('Y True - %s', outNames{j}));
ylabel(sprintf('Y Pred - %s', outNames{j}));
title(sprintf('Parity - %s | R^2=%.3f', outNames{j}, metrics_final.R2(j)));
grid on;
end
sgtitle('Parity plots (True vs. Predicted)');


% Residual histograms
residuals = Y - Yhat_final;
figure('Name','Residual Histograms');
for j = 1:nOut
subplot(ceil(nOut/2),2,j);
histogram(residuals(j,:), 'Normalization','pdf');
title(sprintf('Residuals - %s', outNames{j}));
xlabel('Error'); ylabel('PDF'); grid on;
end
sgtitle('Residuals Histograms');


%% ================== 6) SAVE MODEL ==================
save('NeuroNova_best_model.mat','finalNet','best','tr','inNames','outNames');
fprintf('\nModel saved as: NeuroNova_best_model.mat\n');


%% ================== 7) HOW TO PREDICT NEW CASE ==================
% مثال (بدّل القيم حسب حالتك)
% newCase = struct( ...
% 'T_feed',25,'P_Feed',3000,'EthyleneFeed',0.65,'EthaneFeed',0.07, ...
% 'PropaneFeed',0.18,'Butene1Feed',0.10,'NumberTrays',14,'RefluxRatio',0.34, ...
% 'DistillateRate',9200,'ColumnDiameter',1.5,'ColumnSpacing',0.55, ...
% 'T_Condenser',-4,'T_Reboiler',72,'Duty_Condenser',-950000, ...
% 'Duty_Reboiler',900000,'Area_Cond',70.5,'Area_Reb',2.9);
% y_pred = predict_case(finalNet, newCase, inNames)


fprintf('\nUse predict_case(finalNet, caseStruct, inNames) to get predictions for new inputs.\n');


%% ================== LOCAL FUNCTIONS ==================
function [net, tr] = train_one(X, Y, hl, epochs, goal, showWindow)
net = feedforwardnet(hl, 'trainlm');
net.divideFcn = 'dividetrain'; % الكل تدريب (عدد عينات قليل)
net.trainParam.epochs = epochs;
net.trainParam.goal = goal;
net.trainParam.showWindow = logical(showWindow);
[net, tr] = train(net, X, Y);
end


function [cvMSE, trainMSE, Yhat_cv, R2_cv] = loocv_score(X, Y, hl, epochs, goal, showWindow)
n = size(X,2);
Yhat_cv = zeros(size(Y));
trainMSE_all = zeros(n,1);
for k = 1:n
% قسّم: عيّنة للخارج (test) والباقي تدريب
idx_test = k;
idx_train = setdiff(1:n, k);


Xtr = X(:, idx_train);
Ytr = Y(:, idx_train);
Xte = X(:, idx_test);


[net, ~] = train_one(Xtr, Ytr, hl, epochs, goal, showWindow);
Yhat_cv(:,k) = net(Xte);


% Training MSE لكل fold (على بيانات التدريب الخاصّة به)
Ytr_hat = net(Xtr);
trainMSE_all(k) = mean((Ytr(:)-Ytr_hat(:)).^2);
end


cvMSE = mean((Y(:) - Yhat_cv(:)).^2);
trainMSE = mean(trainMSE_all);


% R2 لكل مخرج
Y_mean = mean(Y, 2);
SS_res = sum((Y - Yhat_cv).^2, 2);
SS_tot = sum((Y - Y_mean).^2, 2);
R2_cv = 1 - SS_res ./ SS_tot;
end


function metrics = compute_metrics(Y, Yhat)
err = Y - Yhat;
mse = mean(err.^2, 2);
rmse = sqrt(mse);
mae = mean(abs(err), 2);
mape = mean(abs(err) ./ max(abs(Y), eps), 2) * 100; % لتفادي القسمة على صفر
% R2
Y_mean = mean(Y,2);
SS_res = sum((Y - Yhat).^2, 2);
SS_tot = sum((Y - Y_mean).^2, 2);
R2 = 1 - SS_res ./ SS_tot;


metrics = struct('MSE', mse, 'RMSE', rmse, 'MAE', mae, 'MAPE', mape, 'R2', R2);
end


function print_metrics(metrics, outNames)
headers = {'Output','MSE','RMSE','MAE','MAPE (%)','R^2'};
fprintf('%-22s | %-10s %-10s %-10s %-10s %-10s\n', headers{:});
fprintf('%s\n', repmat('-',1,80));
for i = 1:numel(outNames)
fprintf('%-22s | %-10.4g %-10.4g %-10.4g %-10.2f %-10.4f\n', ...
outNames{i}, metrics.MSE(i), metrics.RMSE(i), metrics.MAE(i), metrics.MAPE(i), metrics.R2(i));
end
end


function y_pred = predict_case(net, caseStruct, inNames)
% يحوّل struct إلى متجه إدخال بالترتيب الصحيح (inNames)
x = zeros(numel(inNames),1);
for i = 1:numel(inNames)
fn = matlab.lang.makeValidName(inNames{i});
% Map custom names to fields in struct:
% سنستخدم mapping بسيطة:
map = containers.Map( ...
{'T_feed_(°C)','P__Feed_(Kpa)','Ethylene_Feed','Ethane_Feed','Propane_Feed','1-Butene_Feed', ...
'Number_of_Trays','Reflux_Ratio','Distillate_Reat_(Kg/h)','Column_Diameter_(m)', ...
'Column_-Spacing_(m)','T__Condenser_(°C)','T-Reboiler_(°C_)','Duty-Condenser_(K/J)', ...
'Duty-Reboiler_(K/J)','Aera-Condenser_(m2)','Aera-Reboiler_(m2)'}, ...
{'T_feed','P_Feed','EthyleneFeed','EthaneFeed','PropaneFeed','Butene1Feed', ...
'NumberTrays','RefluxRatio','DistillateRate','ColumnDiameter', ...
'ColumnSpacing','T_Condenser','T_Reboiler','Duty_Condenser', ...
'Duty_Reboiler','Area_Cond','Area_Reb'} ...
);


if isKey(map, inNames{i})
sf = map(inNames{i});
else
sf = fn;
end


if ~isfield(caseStruct, sf)
error('Field %s (for input %s) missing in caseStruct.', sf, inNames{i});
end
x(i) = caseStruct.(sf);
end
y_pred = net(x);
end


function [X, Y, inNames, outNames] = buildXY( ...
T_feed, P_Feed, EthyleneFeed, EthaneFeed, PropaneFeed, Butene1Feed, ...
NumberTrays, RefluxRatio, DistillateRate, ColumnDiameter, ColumnSpacing, ...
T_Condenser, T_Reboiler, Duty_Condenser, Duty_Reboiler, Area_Cond, Area_Reb, ...
Top_Ethylene, Top_Ethane, Top_Propene, Top_1Butene, ...
Bot_Ethylene, Bot_Ethane, Bot_Propene, Bot_1Butene)


X = [ T_feed;
P_Feed;
EthyleneFeed;
EthaneFeed;
PropaneFeed;
Butene1Feed;
NumberTrays;
RefluxRatio;
DistillateRate;
ColumnDiameter;
ColumnSpacing;
T_Condenser;
T_Reboiler;
Duty_Condenser;
Duty_Reboiler;
Area_Cond;
Area_Reb ];


Y = [ Top_Ethylene;
Top_Ethane;
Top_Propene;
Top_1Butene;
Bot_Ethylene;
Bot_Ethane;
Bot_Propene;
Bot_1Butene ];


inNames = { ...
'T_feed_(°C)','P__Feed_(Kpa)','Ethylene_Feed','Ethane_Feed','Propane_Feed','1-Butene_Feed', ...
'Number_of_Trays','Reflux_Ratio','Distillate_Reat_(Kg/h)','Column_Diameter_(m)', ...
'Column_-Spacing_(m)','T__Condenser_(°C)','T-Reboiler_(°C_)','Duty-Condenser_(K/J)', ...
'Duty-Reboiler_(K/J)','Aera-Condenser_(m2)','Aera-Reboiler_(m2)'};


outNames = { ...
'Top_Ethylene','Top_Ethane','Top_Propene','Top_1Butene', ...
'Bot_Ethylene','Bot_Ethane','Bot_Propene','Bot_1Butene'};
end


function [T_feed, P_Feed, EthyleneFeed, EthaneFeed, PropaneFeed, Butene1Feed, ...
NumberTrays, RefluxRatio, DistillateRate, ColumnDiameter, ColumnSpacing, ...
T_Condenser, T_Reboiler, Duty_Condenser, Duty_Reboiler, Area_Cond, Area_Reb, ...
Top_Ethylene, Top_Ethane, Top_Propene, Top_1Butene, ...
Bot_Ethylene, Bot_Ethane, Bot_Propene, Bot_1Butene] = embedData()


T_feed = [25 25 25 25 25];
P_Feed = [3000 3000 3000 3000 3000];
EthyleneFeed = [0.6477 0.6257 0.6688 0.6278 0.6620];
EthaneFeed = [0.0691 0.0873 0.0297 0.0882 0.0563];
PropaneFeed = [0.1813 0.1536 0.1898 0.1505 0.2202];
Butene1Feed = [0.1019 0.1334 0.1117 0.1335 0.0615];
NumberTrays = [14 14 14 14 13];
RefluxRatio = [0.342 0.348 0.341 0.349 0.325];
DistillateRate = [9303 8855 8991 8900 9707];
ColumnDiameter = [1.5 1.5 1.5 1.5 1.5];
ColumnSpacing = [0.55 0.55 0.55 0.55 0.55];
T_Condenser = [-3.286 -4.79 -4.141 -4.871 -0.8044];
T_Reboiler = [71.13 74.33 73.38 74.53 61.16];
Duty_Condenser = [-987877.427 -929229.979 -957155.708 -935724.165 -1012492.86];
Duty_Reboiler = [830074.509 1085485.458 976973.014 1073241.787 325247.881];
Area_Cond = [71.3478 69.6872 70.3697 70.2948 70.123];
Area_Reb = [2.3233 3.25925 2.73009 3.22661 0.745241];


Top_Ethylene = [0.7854 0.7832 0.8255 0.7845 0.7680];
Top_Ethane = [0.0761 0.0982 0.0331 0.0988 0.0606];
Top_Propene = [0.1231 0.1000 0.1256 0.0980 0.1603];
Top_1Butene = [0.0154 0.0186 0.0159 0.0187 0.0111];


Bot_Ethylene = [0.2971 0.2621 0.2987 0.2611 0.3770];
Bot_Ethane = [0.0511 0.0622 0.0218 0.0633 0.0446];
Bot_Propene = [0.3297 0.2773 0.3415 0.2734 0.3813];
Bot_1Butene = [0.3321 0.3984 0.3380 0.4021 0.1970];
end



