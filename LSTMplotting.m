clear
clc

% Load time series data with two inputs and one target
load prepared_data.mat

% Load Network
load Trained_LSTM_network.mat

% P means adsoption pressure;
% timead means adsorption time;
% purg means purge to feed ratio;

inputs = [Timead; press; purg];
targets = purity;

% Normalize inputs and targets
[inputs_norm, ps] = mapminmax(inputs);
[targets_norm, ts] = mapminmax(targets);

% Set the size of the training set
trainprop=0.8;
train_size =floor(trainprop*length(targets));

% Separate data into training and test sets
train_inputs = inputs_norm(:, 1:train_size);
train_targets = targets_norm(:, 1:train_size);
test_inputs = inputs_norm(:, train_size+1:end);
test_targets = targets_norm(:, train_size+1:end);

% Simulate the network on the training set
outputs_train_norm = predict(net, train_inputs);

% Simulate the network on the test set
outputs_test_norm = predict(net, test_inputs);

% Denormalize the network outputs
outputs_train = mapminmax('reverse', outputs_train_norm, ts);
outputs_test = mapminmax('reverse', outputs_test_norm, ts);

% Calculate MSE for training set
mse_train = mse(targets(1:train_size), outputs_train);

% Calculate MSE for test set
mse_test = mse(targets(train_size+1:end), outputs_test);


figure
subplot(3,1,1:2)
plot(targets, 'k',LineWidth=1.5)
hold on
plot((outputs_train'),'color','#B0D28A',LineWidth=1.5)


plot(train_size:size(outputs_test,2)+train_size,[outputs_train(end),(outputs_test)]','m',LineWidth=1.5)
ylim([0.88 1])
ylabel({'Purity at the end of Adsorption Step'});
xlabel({'Cycle Number'});
title('\fontsize{18}Network Outputs')
legend('Pseudo-experimental data', 'LSTM output (Training Set)','LSTM Output (Test Set)','Location','southwest')
set(gca,'fontSize',18)

A=abs(outputs_train-targets(1:train_size));
B=abs(outputs_test-targets(train_size+1:end));
C75=[A , B];
subplot(3,1,3)

bar(C75)
ylabel({'Absolute error'});
xlabel({'Cycle Number'});
set(gca,'fontSize',18)

