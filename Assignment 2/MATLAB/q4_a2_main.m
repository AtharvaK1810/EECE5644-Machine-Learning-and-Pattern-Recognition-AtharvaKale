function q4_a2_main(name,nuid,email)


m1=[0 0]; m2=[1 1];
C1=[1 0.5; 0.5 1];
C2=[1 -0.5; -0.5 1];
p1=0.4; p2=0.6;

[X,Y]=meshgrid(linspace(-3,4,300));
Z1=zeros(size(X)); Z2=zeros(size(X));
for i=1:numel(X)
    x=[X(i),Y(i)];
    Z1(i)=log(p1)+log_gaussian_pdf(x,m1,C1);
    Z2(i)=log(p2)+log_gaussian_pdf(x,m2,C2);
end
decision = Z1>Z2;

if ~exist('figs_A2','dir'), mkdir figs_A2; end
figure('Color','w');
contourf(X,Y,decision,[0 1],'LineColor','none'); hold on;
contour(X,Y,Z1-Z2,[0 0],'k','LineWidth',2);
title('A2–Q4: Bayes Decision Boundary (DHS 2.13 Illustration)');
xlabel('x_1'); ylabel('x_2'); grid on;
add_signature(name,nuid,email);
saveas(gcf,'figs_A2/Q4_BayesBoundary.pdf');
end