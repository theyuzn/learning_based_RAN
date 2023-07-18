clc;
clear;
clf;

x = categorical({'Scenario 1','Scenario 2','Scenario 3','Scenario 4','Scenario 5','Scenario 6','Scenario 7'});
x = reordercats(x,{'Scenario 1','Scenario 2','Scenario 3','Scenario 4','Scenario 5','Scenario 6','Scenario 7'});



y = xlsread('/Users/dee/Research/li_sche_pytorch/src/data/result/result_data.xlsx','q_d','b7:h9');

% y_value = [[97 101 93];[97 103 95]; [97 97 90];[95 100 93]; [96 98 92];[97 98 93];[89 92 78]];

b = bar(x,y, '');


% set(gca,'XTick',{'5','10','20','30','40','60','80','120','160','240','320','480', '960', '1920'})
legend('RAS FCFS','RAS DRL-LS','Traditional PF', 'Location', 'northwest', 'interpreter','latex','fontsize', 24, 'NumColumns', 3 );
xlabel('Scenario','interpreter','latex', 'fontsize', 24);
ylabel('Average queuing delay (slot)', 'interpreter','latex','fontsize',24);
set(gca,'FontSize',24);

grid on;