% record = 0;

for k = 14
    
    load(['propagation_data_' num2str(k) '.mat']);
    
    if k==11
        M = 700;
    end
    
    for i = 1:1:M
        disp(i);
        figure(1)
        clf
        subplot(3,2,1:4);
        hold on
        % plot(X(:,1),-X(:,2),'.c');
        plot(Y{i}(:,1),-Y{i}(:,2),'.r');
        plot(m(1:i,1),-m(1:i,2),'-k');
        plot(m(i,1),-m(i,2),'ok','markerfacecolor','b');
        hold off
        axis equal
        axis([-171.0700  293.8600 176.4200 430.5800 ]);
        title('Object position');
        
        
        subplot(3,2,5);
        histfit(Y{i}(:,1),100);
        title('x axis');
        ylim([0 15]);
        subplot(3,2,6);
        histfit(Y{i}(:,2),100);
        title('y axis');
        ylim([0 15]);
        
%         drawnow;
        
        if record
            frame = getframe(gcf);
            for k = 1:5
                writeVideo(writerObj, frame);
            end
        end
        
    end
    drawnow;
    hold off
    
end