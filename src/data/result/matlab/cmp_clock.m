clc;
clear;
clf;

x = categorical({'100','200','300'});
x = reordercats(x,{'100','200','300'});



y = xlsread('/Users/dee/Research/li_sche_pytorch/src/data/result/result_data.xlsx','clock','h9:i11');


% 176	364	1098
% 76	128	183
		
y_c = [176 364 1098];
y_g = [76 128 183];

b = bar(x,y, '');
b(1).FaceColor = [0.9290 0.6940 0.1250];
hold on;
l = plot(x,smoothdata(y_c,"gaussian",3), x,smoothdata(y_g,"gaussian",3));
l(1).LineWidth = 3;
l(1).Color = [0.9290 0.6940 0.1250];
l(2).LineWidth = 3;
l(2).Color = 'red';



% set(gca,'XTick',{'5','10','20','30','40','60','80','120','160','240','320','480', '960', '1920'})
legend('PF','RAS DRL-LS', 'Location', 'northwest', 'interpreter','latex','fontsize', 24, 'NumColumns', 3 );
xlabel('$N_{UE}$','interpreter','latex', 'fontsize', 24);
ylabel('Computation time ($us$)', 'interpreter','latex','fontsize',24);
set(gca,'FontSize',24);

grid on;