clear %all
clc
%% initialize
% objective function
U = @(x) 4 + .5*x - 1.2*x.^2 + .12*x.^4;
gU = @(x) .5 - 2.4*x + .48*x.^3; % grad U, must be updated along with U

% Random sample options
UseMonteCarlo = false;
d = 10; % sample size

% SDE options
T = 200; % maximum time; will be truncated if Beta increases rapidly
h = 1e-5; % time step length
Beta0 = .001;
%Beta_dot0 = 0.1;

Beta_dot0 = 0.1 * Beta0;
Zeta0 = 1;

% plot options
Bin_Edges = [-10, linspace(-4,4,21), 10]; % histogram bin edges

% other constants
N = T/h;

%% get random samples
MCMC = MCMCRandGen;
if UseMonteCarlo
    MCMC.beta = Beta0;
    MCMC.U = U;
    % A.nsigma = 800;
    MCMC.sigma = 2;
    MCMC.s0 = 2;
    MCMC.n_dec = 73; % decorrelation distance, ~100
    MCMC.Nrand = ceil(d/10); % batch gaussian random generator size
    MCMC.resetRand % initialize random generator
    x_initial = MCMC.GetNSt(d); % get N states
else % simple samples
    x_initial = 3 * ones(d,1);
end

t = 0:h:T; % time vector [0 1h 2h 3h ... Nh]
x = zeros(d,length(t));
x(:,1)=transpose(x_initial); % initial height

%% solve SDE
Beta = zeros(N,1);
Beta(1) = Beta0;
Beta_dot = zeros(N-1,1);
Beta_dot_cur = Beta_dot0;
Zeta = Zeta0; % to be used in the future
MCMC.updateProgressBar(-1,1);
for ii=1:N
%     Step 1: update SDE
    Beta_dot(ii) = Beta_dot_cur;
    Beta_cur = Beta(ii) + Beta_dot_cur * h;
    Beta(ii+1) = Beta_cur;
    xj = x(:,ii); % x(t)
    dx = -gU(xj);
    xi = x(:,ii) + dx.*h + sqrt(2*h/Beta_cur)*randn(d,1); % x(t+dt)
    x(:,ii+1) = xi;
    
%     Step2: estimate rho
    Y = repmat(-dx.',d,1);
    [Xi,Xj] = ndgrid(xi,xj);
    V = Xi - Xj + 1/Zeta0 * Y * h;
    eV2 = exp(-V.^2/ 2/h*Beta_cur);
    Zdenom = sum(eV2, 2);
    Znumer = sum(-V .* eV2, 2) / h * Beta_cur;
    Z = Znumer ./ Zdenom;
    
    W = gU(xi);
    A = Zeta0^-2 * mean(W.^2) - Beta_cur^-2 * mean(Z.^2);
    
    B = mean(xi.^2);
    C = mean(xi .* W);
    
%     Step3: Based on the rho, update Beta_dot by E-L equation
    Beta2dot_cur = Beta_dot_cur^2/2/Beta_cur + 2*Beta_dot_cur/Zeta0 * C / B ....
        - 2*Beta_dot_cur / B / Beta_cur + 2 * Beta_cur / B * A;
    Beta_dot_cur = Beta_dot_cur + Beta2dot_cur * h;
    if Beta_dot_cur <= 0 % this causes ocsillation of Beta_dot
       % Beta_dot_cur = Beta_dot0;
       Beta_dot_cur = Beta_cur*0.1;
    end
    if mod(ii, ceil(N*.01))==0
        MCMC.updateProgressBar(ii,N);
    end
    if Beta2dot_cur > 100 || Beta_cur > 1e+2 %Beta_dot_cur > 1 % truncate simulation
        N = ii;
        Beta_dot = Beta_dot(1:(N-1));
        Beta = Beta(1:N);
        x = x(:,1:N);
        t = t(1:N);
        T = h * N;
        MCMC.updateProgressBar(1,1);
        warning('Beta is increasing too fast. Step %d, Time %f', N, T)
        break
    end
end

%% plot samples
figure
% trace
plot(t,x)
% U
hold on
H1.ux = -linspace(-4,4,101);
H1.uy = -U(H1.ux);
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

hold off

axis([-T*.25,T*1.25,-10,10])   % set axis limits
grid on
title('Result')

%% plot beta
figure
[H2.hx,H2.hp1,H2.hp2] = plotyy(t, Beta, t(1:end-1),Beta_dot);
H2.hp1.DisplayName = '$\beta$';
H2.hp2.DisplayName = '$\dot{\beta}$';
H2.hl = legend([H2.hp1,H2.hp2], 'Location','northwest');
H2.hl.Interpreter = 'latex';
ylim(H2.hx(1), [-.3,1.2] * max(Beta))
ylim(H2.hx(2), [0,5*max(Beta_dot)])