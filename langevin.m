%%
% This is method II


clear %all
clc
%% initialize
% objective function
U = @(x) 4 + .5*x - 1.2*x.^2 + .12*x.^4;
gU = @(x) .5 - 2.4*x + .48*x.^3; % grad U, must be updated along with U

d = 100; % sample size

% SDE options
T = 150; %maximum time; will be truncated if Beta increases rapidly
h = 1e-4;

% plot options
Bin_Edges = [-10, linspace(-4,4,21), 10]; % histogram bin edges

% other constants
N = T/h;

%%
x_initial = 3 * ones(d,1);
t = 0:h:T; % time vector [0 1h 2h 3h ... Nh]
x = zeros(d,length(t));
y = zeros(d,length(t));
y(:,1) = 1;
avg_1 = zeros(1,length(t));%delx_Uy_avg
avg_2 = zeros(1,length(t));%y_sqr_avg
avg_3 = ones(1,length(t));%dely_rho_avg
x(:, 1) = 3;
beta = zeros(1,length(t));
beta_dot = zeros(1,length(t));
beta2dot = zeros(1,length(t));
beta_cur = 0.001;
beta_dot_cur = beta_cur/10;
beta(1) = beta_cur;
for ii = 1:N
    %% Step 1
    beta_dot(ii) = beta_dot_cur;
    beta_cur = beta(ii) + beta_dot_cur * h;
    beta(ii+1) = beta_cur;
    
    yj = y(:, ii);
    dy =  - gU(x(:, ii)) - y(:, ii);
	%y(:,ii+1) = y(:,ii) + (-gU(x(:,ii)) - y(:,ii))*h + sqrt(2*h/beta_cur)*randn(d,1);
    y(:, ii + 1) = y(:, ii) + dy.*h + sqrt(2 * h /beta_cur) * randn(d, 1);
	x(:,ii+1) = x(:,ii) + y(:,ii).*h;
    yi = y(:, ii +1);
    
    %% step 2 find average and approximate \rho 
    avg_1_cur = mean(gU(x(:,ii)).*y(:,ii));
    %avg_1(ii) = avg_1_cur;
    avg_2_cur = mean(y(:,ii).^2);
    %avg_2(ii) = avg_2_cur;
    %% Let us estimate rho (the partial derivative here).
    %avg_3_cur = 1;
    Y = repmat(-dy.', d, 1);
    
    [Yi, Yj] = ndgrid(yi, yj);
    
    V = Yi - Yj + Y * h;
    
    eV2 = exp(-V.^2/ 2 / h * beta_cur);
    
    Zdenom = sum(eV2, 2);
    
    Znumer = sum(-V .* eV2, 2) / h * beta_cur;
    
    Z = Znumer ./ Zdenom;
    
    avg_3_cur = mean(Z.^2);
    %avg_3(ii) = avg_3_cur;
    
    % step 3 Based on the rho, update Beta_dot
    beta2dot_cur = beta_dot_cur^2 / 2 / beta_cur ...
        + 2*beta_dot_cur*(avg_1_cur/avg_2_cur - 1/avg_2_cur +1) ...
        + 2*beta_cur - 2*avg_3_cur/beta_cur/avg_2_cur;
     %beta2dot_cur = 0;
    beta_dot_cur = beta_dot_cur + beta2dot_cur * h;
    if beta_dot_cur <= 0 % this causes ocsillation of Beta_dot
        beta_dot_cur = 0.1  * beta_cur; %beta_dot(1);
    end
    if beta_dot_cur > 10
        N = ii;
        beta_dot = beta_dot(1:(N-1));
        beta = beta(1:N);
        x = x(:,1:N);
        t = t(1:N);
        T = h * N;
        warning('Beta is increasing too fast. Step %d, Time %f', N, T)
        break
    end
end
%% plot
sz = size(x);
x_amt = sz(1);
figure
hold on
plot(t,x)

% orignial U on the left
H1.ux = -linspace(-4,4,101);
H1.uy = U(H1.ux);
H1.uy = (H1.uy - min(H1.uy))/(max(H1.uy)-min(H1.uy)); % normalization
plot(-.02*T -.2*T * H1.uy, H1.ux) % place U(x) on the left 25% space

% histogram
% final distribution
H1.y_final = histcounts(x(:,end),Bin_Edges);
H1.y_final = H1.y_final / max(H1.y_final);
stairs(1.02* T+.2*T * H1.y_final([1,1:end]),Bin_Edges)
% initial distribution
H1.y_initial = histcounts(x_initial,Bin_Edges);
H1.y_initial = H1.y_initial / max(H1.y_initial);
stairs(-.02*T -.2*T * H1.y_initial([1,1:end]),Bin_Edges)
grid on