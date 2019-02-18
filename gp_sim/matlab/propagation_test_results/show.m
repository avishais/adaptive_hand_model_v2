clear all

record = 1;

if record
    writerObj = VideoWriter(['propagation_15.avi']);
    writerObj.FrameRate = 60;
    open(writerObj);
end

%%

for k = 15%1:14
    
    load(['propagation_data_' num2str(k) '.mat']);
    
    if k==11
        M = 700;
    end
    
    for i = 1:M
        disp(i);
        figure(1)
        clf
%         subplot(121);
        hold on
%         plot(X(:,1),-X(:,2),'.c');
        plot(Y{i}(:,1),-Y{i}(:,2),'.r');
        plot(m(1:i,1),-m(1:i,2),'-k');
        plot(m(i,1),-m(i,2),'ok','markerfacecolor','b');
        hold off
        axis equal
        axis([-171.0700  293.8600 176.4200 430.5800 ]);
        title('Object position');
        
%         subplot(122);
%         hold on
%         % plot(X(:,1),-X(:,2),'.c');
%         plot(Y{i}(:,3),Y{i}(:,4),'.r');
%         plot(m(1:i,3),m(1:i,4),'-k');
%         plot(m(i,3),m(i,4),'ok','markerfacecolor','b');
%         hold off
%         axis equal
%         axis([-163  601 -556 16]);
%         title('Gripper load');
        
        drawnow;
        
        if record
            frame = getframe(gcf);
            writeVideo(writerObj, frame);
        end
        
    end
    drawnow;
    hold off
    
end

%%
% for k = 11
%     % k = 11;
%     load(['propagation_data_' num2str(k) '.mat']);
%     M = 700;
%     figure(1)
%     clf
%     axis equal
%     axis([-171.0700  293.8600 176.4200 430.5800 ]);
%     hold on
%     for i = 2:M
%         for j = 1:N
%             plot([Y{i-1}(j,1) Y{i}(j,1)],-[Y{i-1}(j,2) Y{i}(j,2)],'-r');
%         end
%         plot(m(1:i,1),-m(1:i,2),'-k');
%         drawnow;
%         
%         if record
%             frame = getframe(gcf);
%             writeVideo(writerObj, frame);
%         end
%     end
%     hold off
% end
%%

if record
    close(writerObj); % Saves the movie.
end