function [idx, C, sumD, D] = MyKmeans(X, K, C0, numIter)

%%%%%%%%%%%%%%%%Input%%%%%%%%%%%%%%%%%%
%X is data
%K is num clusters
%C0 is initial cluster centers
%numIter is num iterations for function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

numRowX = length(X(:,1));
numColX = length(X(1,:));

%%%%%%%%%%%%%%%%Output%%%%%%%%%%%%%%%%%
%idx is cluster assignment for each xj
%C is cluster centroid locations
%sumD is within cluster sums of point to centroid distances
%D is distance from each point xj to every centroid
idx = zeros(numRowX,numIter);
C = C0;
sumD = zeros(K,numIter);
D = zeros(numRowX,K);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Dist = zeros(1,K);

%Perform numIter iterations for the k-means algorithm
%i is current iteration
for i = 1:numIter
    
    %For each data point, calculate its distance to each centroid
    %Choose the nearest centroid
    %j is index of x_j
    for j = 1:numRowX
        
        %Calculate dist for each centroid
        %k is index of C_k
        for k = 1:K
            Dist(k) = sum(power(abs(X(j,:) - C(k,:)), 2));     
        end
        
        %Set x_j to be a member of the cluster with the nearest center
        [~, idx(j,i)] = min(Dist);
    end
    %All points x_j have now been assigned to nearest cluster
    
    %Calculate sumD for current iteration
    for j = 1:numRowX
        cidx = idx(j,i);
        sumD(cidx,i) = sumD(cidx,i) + sum(power(abs(X(j,:) - C(cidx,:)), 2));
    end
    
    %Recalculate cluster centers
    C = zeros(K, numColX);    
    %Take sum of distances to centroid per cluster and divide by number 
    %of x_j in that cluster to find new cluster center
    for j = 1:numRowX
        cidx = idx(j,i);
        C(cidx,:) = C(cidx,:) + X(j,:);
    end
    
    for k = 1:K
        C(k,:) = C(k,:)./numel(find(idx(:,i)==k));
    end
    %New cluster centers have been calculated, proceed to next iteration
end

%K-means algorithm is complete
%Collect data and return
%Calculate D
%For each x_j in X calculate its distance to each cluster C_k center 
%and store in D
for j = 1:numRowX  
    for k = 1:K
        D(j,k) = sum(power(abs(X(j,:) - C(k,:)), 2));
    end
end

end


