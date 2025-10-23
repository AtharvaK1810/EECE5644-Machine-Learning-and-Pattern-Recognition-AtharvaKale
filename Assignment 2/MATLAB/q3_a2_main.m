function out = q3_a2_main(name,nuid,email)

rng(11,'twister');

sigx=0.25; sigy=0.25; Sig_inv=diag([1/sigx^2, 1/sigy^2]);
sigma_r=0.3;
theta=2*pi*rand; r=0.8*sqrt(rand); xT=[r*cos(theta); r*sin(theta)];

Ks=1:4; [Xg,Yg]=meshgrid(linspace(-2,2,250));
allVals=[];

% Gather global contour levels
for K=Ks
    L=landmarks(K);
    r_true=sqrt(sum((xT'-L).^2,2));
    r_meas=abs(r_true + sigma_r*randn(K,1));
    Z=zeros(size(Xg));
    for i=1:numel(Xg)
        x=[Xg(i);Yg(i)];
        prior=0.5*(x'*Sig_inv*x);
        d=sqrt(sum((x'-L).^2,2));
        like=sum((r_meas-d).^2)/(2*sigma_r^2);
        Z(i)=prior+like;
    end
    allVals=[allVals; Z(:)];
end
levels = quantile(allVals, linspace(0.05,0.95,10));

if ~exist('figs_A2','dir'), mkdir figs_A2; end
for K=Ks
    L=landmarks(K);
    r_true=sqrt(sum((xT'-L).^2,2));
    r_meas=abs(r_true + sigma_r*randn(K,1));
    Z=zeros(size(Xg));
    for i=1:numel(Xg)
        x=[Xg(i);Yg(i)];
        prior=0.5*(x'*Sig_inv*x);
        d=sqrt(sum((x'-L).^2,2));
        like=sum((r_meas-d).^2)/(2*sigma_r^2);
        Z(i)=prior+like;
    end
    figure('Color','w');
    contour(Xg,Yg,Z,levels,'LineWidth',1.1); axis equal tight; hold on;
    plot(L(:,1),L(:,2),'ko','MarkerFaceColor','w');
    plot(xT(1),xT(2),'r+','MarkerSize',10,'LineWidth',1.5);
    title(sprintf('A2–Q3 — MAP Objective Contours (K=%d)',K));
    xlabel('x'); ylabel('y'); grid on;
    add_signature(name,nuid,email);
    saveas(gcf, sprintf('figs_A2/Q3_contours_K%d.pdf',K));
end
out=struct();
end

function L=landmarks(K)
ang=linspace(0,2*pi,K+1)'; ang(end)=[];
L=[cos(ang) sin(ang)];
end