
clc
clear all
close all

% Load timeseries data with two inputs and one target
load prepared_data.mat
train_size=720;

% P means adsoption pressure;
% timead means adsorption time;
% purg means purge to feed ratio;

inputs = [Timead; press; purg];
[inputs_norm, ps] = mapminmax(inputs);

targets = purity;
[target_norm, ts] = mapminmax(purity);

% Separate data into training and test sets
train_inputs = inputs_norm(:, 1:train_size);
train_targets = target_norm(:, 1:train_size);
test_inputs = inputs_norm(:, train_size+1:end);
test_targets = target_norm(:, train_size+1:end);

% Create NARX network
input_delays = 1:1;
feedback_delays = 1:1;
hidden_layer_size = 20;
narx_net = narxnet(input_delays, feedback_delays, hidden_layer_size);

% Prepare training set data
[Xs,Xi,Ai,Ts] = preparets(narx_net, con2seq(train_inputs), {}, con2seq(train_targets));

% Prepare test set data
[Xs_test,Xi_test,Ai_test,Ts_test] = preparets(narx_net, con2seq(test_inputs), {}, con2seq(test_targets));


tic
narx_net = trainlm(narx_net, Xs, Ts, Xi, Ai); %trainbr %trainscg
toc

narx_net = closeloop(narx_net);

% Prepare training set data
[Xs,Xi,Ai,Ts] = preparets(narx_net, con2seq(train_inputs), {}, con2seq(train_targets));

% Prepare test set data
[Xs_test,Xi_test,Ai_test,Ts_test] = preparets(narx_net, con2seq(test_inputs), {}, con2seq(test_targets));

tic
% Simulate network on training set

outputs_train = narx_net(Xs, Xi,Ai);

% Simulate network on test set

outputs_test = narx_net(Xs_test, Xi_test,Ai_test);

toc

outputs_train = mapminmax('reverse', cell2mat(outputs_train), ts);
outputs_test = mapminmax('reverse', cell2mat(outputs_test), ts);

Ts= mapminmax('reverse', cell2mat(Ts), ts);
Ts_test= mapminmax('reverse', cell2mat(Ts_test), ts);

mse_train = mse(targets(1:train_size-1), outputs_train)
mse_test = mse(targets(train_size+2:end), outputs_test)


figure
subplot(3,1,1:2)
% plot((Ts)', 'k',LineWidth=2)
hold on
plot((targets)', 'k',LineWidth=1.5)
plot((outputs_train)', 'color','#B0D28A',LineWidth=1.5)
% plot(train_size+1:size(outputs_test,2)+train_size, (Ts_test)','k',LineWidth=2)
plot(train_size+1:size(outputs_test,2)+train_size, (outputs_test)','m',LineWidth=1.5)
ylabel({'Purity at the end of Adsorption Step'});
% title({'\fontsize{12}Network Outputs'; ['Training set MSE = ' num2str(mse_train)];['Test set MSE = ' num2str(mse_test)]})
% legend('Target (Training Set)', 'Output (Training Set)', 'Target (Test Set)', 'Output (Test Set)','Location','southeast')
legend('Pseudo-experimental data','NARX Output (Training Set)','NARX Output (Test Set)','Location','southwest')
set(gca,'fontSize',13)
% 
% 

A=abs(outputs_train-Ts);
B=abs(outputs_test-Ts_test);
Cnx=[A , B];
subplot(3,1,3)
bar(Cnx)
ylabel({'Absolute error'});
xlabel({'Cycle Number'});
set(gca,'fontSize',13)

