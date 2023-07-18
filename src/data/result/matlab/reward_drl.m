clc;
clear;
clf;

episode = xlsread('/Users/dee/Research/li_sche_pytorch/src/data/result/result_data.xlsx','DRL-LS_reward','f2:f71');

scenario_1 = xlsread('/Users/dee/Research/li_sche_pytorch/src/data/result/result_data.xlsx','DRL-LS_reward','a2:a71');
scenario_2 = xlsread('/Users/dee/Research/li_sche_pytorch/src/data/result/result_data.xlsx','DRL-LS_reward','b2:b71');
scenario_3 = xlsread('/Users/dee/Research/li_sche_pytorch/src/data/result/result_data.xlsx','DRL-LS_reward','c2:c71');
scenario_4 = xlsread('/Users/dee/Research/li_sche_pytorch/src/data/result/result_data.xlsx','DRL-LS_reward','d2:d71');
scenario_5 = xlsread('/Users/dee/Research/li_sche_pytorch/src/data/result/result_data.xlsx','DRL-LS_reward','e2:e71');


scenario_1 = smoothdata(scenario_1,"gaussian",5);
scenario_2 = smoothdata(scenario_2,"gaussian",5);
scenario_3 = smoothdata(scenario_3,"gaussian",5);
scenario_4 = smoothdata(scenario_4,"gaussian",5);
scenario_5 = smoothdata(scenario_5,"gaussian",5);


plot(episode,scenario_1,'LineWidth',2);
hold on;
plot(episode,scenario_2,'LineWidth',2);
hold on;
plot(episode,scenario_3,'LineWidth',2);
hold on;
plot(episode,scenario_4,'LineWidth',2);
hold on;
plot(episode,scenario_5,'LineWidth',2);
hold off;

% set(gca,'XTick',{'5','10','20','30','40','60','80','120','160','240','320','480', '960', '1920'})
legend('Scenarion1','Scenarion2','Scenarion3','Scenarion4','Scenarion5', 'Location', 'northwest', 'interpreter','latex','fontsize', 18, 'NumColumns', 1 );
xlabel('Episodes','interpreter','latex', 'fontsize', 18);
ylabel('Cumulative Reward in Each Episode', 'interpreter','latex','fontsize',18);
set(gca,'FontSize',18);

grid on;