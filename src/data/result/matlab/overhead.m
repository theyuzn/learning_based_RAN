clc;
clear;
clf;

x = categorical({'100','200','300'});
x = reordercats(x,{'100','200','300'});



y = xlsread('/Users/dee/Research/li_sche_pytorch/src/data/result/result_data.xlsx','overhead','d2:f4');


% 5920	18302	54908
% 4930	12576	24798
		
y_c = [5920 18302 54908];
y_g = [4930 12576 24798];

b = bar(x,y, '');
hold on;
l = plot(x,smoothdata(y_c,"gaussian",3), x,smoothdata(y_g,"gaussian",3));
l(1).LineWidth = 3;
l(1).Color = 'blue';
l(2).LineWidth = 3;
l(2).Color = 'red';



% set(gca,'XTick',{'5','10','20','30','40','60','80','120','160','240','320','480', '960', '1920'})
legend('FCFS','DRL-LS', 'Location', 'northwest', 'interpreter','latex','fontsize', 24, 'NumColumns', 3 );
xlabel('$N_{UE}$','interpreter','latex', 'fontsize', 24);
ylabel('Average overhead (bits)', 'interpreter','latex','fontsize',24);
set(gca,'FontSize',24);

grid on;