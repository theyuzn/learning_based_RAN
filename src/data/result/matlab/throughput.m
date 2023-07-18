clc;
clear;
clf;

x = categorical({'Scenario 1','Scenario 2','Scenario 3','Scenario 4','Scenario 5','Scenario 6','Scenario 7'});
x = reordercats(x,{'Scenario 1','Scenario 2','Scenario 3','Scenario 4','Scenario 5','Scenario 6','Scenario 7'});

scenario_1 = xlsread('/Users/dee/Research/li_sche_pytorch/src/data/result/result_data.xlsx','tp','b3:b5');
scenario_2 = xlsread('/Users/dee/Research/li_sche_pytorch/src/data/result/result_data.xlsx','tp','c3:c5');
scenario_3 = xlsread('/Users/dee/Research/li_sche_pytorch/src/data/result/result_data.xlsx','tp','d3:d5');
scenario_4 = xlsread('/Users/dee/Research/li_sche_pytorch/src/data/result/result_data.xlsx','tp','e3:e5');
scenario_5 = xlsread('/Users/dee/Research/li_sche_pytorch/src/data/result/result_data.xlsx','tp','f3:f5');
scenario_6 = xlsread('/Users/dee/Research/li_sche_pytorch/src/data/result/result_data.xlsx','tp','g3:g5');
scenario_7 = xlsread('/Users/dee/Research/li_sche_pytorch/src/data/result/result_data.xlsx','tp','h4:h5');

y = xlsread('/Users/dee/Research/li_sche_pytorch/src/data/result/result_data.xlsx','tp','b3:h8');

% y_value = [[97 101 93];[97 103 95]; [97 97 90];[95 100 93]; [96 98 92];[97 98 93];[89 92 78]];

b = bar(x,y, '');


% set(gca,'XTick',{'5','10','20','30','40','60','80','120','160','240','320','480', '960', '1920'})
legend('RAS FCFS $N_{UE} = 100$','RAS FCFS $N_{UE} = 200$','RAS FCFS $N_{UE} = 300$', ...
        'RAS DRL-LS $N_{UE} = 100$', 'RAS DRL-LS $N_{UE} = 200$', 'RAS DRL-LS $N_{UE} = 300$', ...
        'Location', 'northwest', 'interpreter','latex','fontsize', 24, 'NumColumns', 1 );
xlabel('Scenario','interpreter','latex', 'fontsize', 24);
ylabel('System throughput (Mb/s)', 'interpreter','latex','fontsize',24);
set(gca,'FontSize',24);

grid on;