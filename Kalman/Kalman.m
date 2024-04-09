% M126 Winter 2020, Data-driven UQ
% by Yoonsang Lee (yoonsang.lee@dartmouth.edu)
%
% Lecture 17: Kalman Filter
%
% Kalman filtering for discrete 2x2 linear systems
%
% Note: this code is not optimized,
%       for example, this code uses inv()
%       Do not use this code for heavy applications
%       Teaching purpose only
%
% Last update: Feb 21, 2020
rng(1) % seed for random number generator

theta=.3; % rotation angle
A=[cos(theta),-sin(theta);sin(theta),cos(theta)]; % transition(rotatino) matrix

u0=[5;0]; % initial value for u

k=100; % total time to solve for u

u=zeros(2,k); % memory for u; true value
v=zeros(2,k); % memory for v; observation


u(:,1)=u0; v(:,1)=u0;


sigma=.1; % noise level
sigma0=2; % observation noise level

C=[sigma0^2,0;0,sigma0^2];
H=[1,0;0,1];

% Generate the true values and observations
for i = 1:k-1
    u(:,i+1) = A*u(:,i)+sigma*randn(2,1);
    v(:,i+1) = H*u(:,i+1)+sigma0*randn(2,1);
    
    if(i<15)
        figure(1)
        plt_soln(u,v,i);
        drawnow();
        input('press any keys >>')
    end
end

figure(1)
plt_soln(u,v,k);
input('press any keys to use Kalman Filter >>')

%% Kalman Filtering; observation v is given in the previous part.
% find the posterior estimate of u, uest
uest=zeros(2,k); % posterior mean of u, i.e., estimate of u using the posterior distribution
uest(:,1)=u0+[1;-1];

for i = 1:k-1
    uprior = A*uest(:,i);
    Ctmp = A*C*A'+sigma^2*eye(2);
    
    K = Ctmp*H'*inv(H*Ctmp*H'+sigma0^2*eye(2));
    
    C = (eye(2)-K*H)*Ctmp;
    uest(:,i+1) = uprior + K*(v(:,i+1)-H*uprior);
    
    if(i<30)
        figure(2)
        plt_filtered_soln(u,v,uest,i);
        drawnow();
        input('press any keys >>')
    end
end

figure(2);
plt_filtered_soln(u,v,uest,k);

