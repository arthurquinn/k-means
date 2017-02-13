function TestMyKmeans(filename, numRepeat, numIter)

data = importdata(filename);
Y = data(:,1)+1;
X = data(:,2:end);
clear data;
K = max(Y);
n = length(Y);

%%% Normalize the data to have unit L2 norm
temp = sqrt( sum( X.^2, 2 ) );
temp( temp == 0 ) = 1;
X = bsxfun( @rdivide, X, temp );
clear temp;
%%% End normalize


SD1 = zeros(numRepeat,numIter);

for i = 1:numRepeat

C0 = X(randsample(n,K),:);
tic;[idx1{i},C1,sumd1,D1{i}]=MyKmeans(X,K,C0,numIter); T1(i) = toc;
tic;[idx2{i},C2,sumd2,D2{i}]=kmeans(full(X),K,'Start',full(C0),'Maxiter',numIter);
T2(i)=toc;
SD1(i,:) = sum(sumd1,1);
SD2(i,:) = sum(sumd2);


for t = 1:numIter
    acc1(i,t) = evalClust_Error(idx1{i}(:,t),Y);
end

%%% Evaluate the classification accuracy for Maltab Kmeans
acc2(1,i) = evalClust_Error(idx2{i},Y);
%%% End evaluation

output = [acc1(1:i,end) acc2(1:i)' SD1(1:i,end) SD2(1:i) T1(1:i)' T2(1:i)'];
feval('save',[filename '.summary.txt'],'output','-ascii');
end

%%% Plot Iterations vs SD
figure;
plot(1:numIter,SD1,'linewidth',1);hold on; grid on;
set(gca,'FontSize',20);
xlabel('Iteration');ylabel('SD');
title(filename);
%%% End Plot Iterations vs SDS
%%% Plot accuracies
figure;
plot(1:numIter,acc1,'linewidth',1);hold on; grid on;
set(gca,'FontSize',20);
xlabel('Iteration');ylabel('Accuracy (%)');
title(filename);
%%% End Plot accuracies
%%% Plot Times
figure;
plot(T1,T2,'r.');hold on;grid on;
set(gca,'FontSize',20);
xlabel('MyKmeans Time');ylabel('MATLAB kmeans time');
title(filename);
%%% End Plot Times

end