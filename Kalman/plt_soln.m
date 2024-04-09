function plt_soln(u, v, ind)
if(ind<15)
    xmax=15;
else
    xmax=ind;
end
subplot(2,2,1)
plot(u(1,1:ind),u(2,1:ind),'-d')
title('Trajectory of true values')
xlim([-10,10])
ylim([-10,10])

subplot(2,2,2)
plot(v(1,1:ind),v(2,1:ind),'-d')
title('Trajectory of observation')
xlim([-10,10])
ylim([-10,10])

subplot(2,2,3)
plot(1:ind,u(1,1:ind),1:ind,v(1,1:ind),'-+')
title('Time series of u_1')
legend('true','observation')
ylim([-10,10])
xlabel('time')
ylabel('u_1')
ylim([-10,10])
xlim([1 xmax])

subplot(2,2,4)
plot(1:ind,u(2,1:ind),1:ind,v(2,1:ind),'-+')
title('Time series of u_2')
legend('true','observation')
ylim([-10,10])
xlabel('time')
ylabel('u_2')
ylim([-10,10])
xlim([1 xmax])
end