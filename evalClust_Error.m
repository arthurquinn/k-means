function [acc, matchV] = evalClust_Error( o, y )
% [err, matchV] = ECL_Error( o, y )
%
% Compute the clustering error
% with the bipartite maximal weighted matching algorithm
%
% Mingrui Wu, 2006

% build edge matrix
nData = length( y );
nC = max( y );
E = zeros( nC, nC );
for m = 1 : nData
    i1 = o( m );
    i2 = y( m );
    E( i1, i2 ) = E( i1, i2 ) + 1;
end
% E = int32( E' );
n_clusters = length(unique(y));
for i=1:n_clusters
    c(i) = length(find(y==i));
end
C = repmat(c,n_clusters,1);
E = C-E;
[assign,eMatch]=hungarian(E);

matchV = nData-eMatch;
acc = matchV/nData;