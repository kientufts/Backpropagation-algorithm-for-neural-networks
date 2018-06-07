function [fig1,fig2]=neuralNet(d,w,dataName,dataTest,iters)
    eta=.1; rand('seed', 0)
    dat = importdata(strcat('./pp4data/',dataName,'.arff'));
    datTest= importdata(strcat('./pp4data/',dataTest,'.arff'));
    txt=dat.textdata;
    txt=char(txt(length(txt)-1));
    noLabels=1;
    for idx=1:length(txt)
        if txt(idx)==','
            noLabels=noLabels+1;
        end
    end
    data=dat.data;
    dataTest=datTest.data;
    [m, n]=size(data);
    [m1, n1]=size(dataTest);
    % m: number of examples
    % n-1: number of inputs
    lw=(d-1)*w+n-1+noLabels;
    weights = 0.2*rand(lw,w)-.1;
    sh=zeros(d,w); hid=zeros(d,w); sigmah=zeros(d,w); deltah=zeros(d,w);
    so=zeros(1,noLabels); out =zeros(1,noLabels); sigmao=zeros(1,noLabels); deltao=zeros(1,noLabels);
    train_error=zeros(1,iters); test_error=zeros(1,iters);
    for idx=1:iters
         % For each example in traindata
         error=0;
         for idx2=1:m
             % set the desired value
             desireo=zeros(1,noLabels);
             desireo(data(idx2,n)+1)=1;
             % Update weights using backpropagation formulas
             % Calculate values for the first hidden layer
             for idx3=1:w
                 sh(1,idx3)=data(idx2,1:n-1)*weights(1:n-1,idx3);
                 hid(1,idx3)=1/(1+exp(-sh(1,idx3)));
                 sigmah(1,idx3)=hid(1,idx3)*(1-hid(1,idx3));
             end
             %Calculate values for the remaining hidden layer
             for idx3=2:d
                 for idx4=1:w
                     sh(idx3,idx4)=hid(idx3-1,:)*weights(n+(idx3-2)*w:n-1+(idx3-1)*w,idx4);
                     hid(idx3,idx4) = 1/(1+exp(-sh(idx3,idx4)));
                     sigmah(idx3,idx4) = hid(idx3,idx4)*(1-hid(idx3,idx4));
                 end
             end
             % Calculate values for the output layer
             for idx3=1:noLabels
                 so(idx3)=hid(d,:)*weights(n-1+(d-1)*w+idx3,:)';
                 out(idx3)=1/(1+exp(-so(idx3)));
                 sigmao(idx3)=out(idx3)*(1-out(idx3));
             end
             [ma, id]=max(out);
             if abs(id-data(idx2,n)-1)>0
                 error=error+1;
             end
             % compute delta
             for idx3=1:noLabels
                 deltao(idx3)=-sigmao(idx3)*(desireo(idx3)-out(idx3));
             end
             for idx3=1:w % compute delta for the last hidden layer
                 deltah(d,idx3)=sigmah(d,idx3)*deltao*weights(n+(d-1)*w:lw,idx3);
             end
             for idx3=d-1:-1:1 % compute delta for the remaining hidden layer
                 for idx4=1:w
                     deltah(idx3,idx4)=sigmah(idx3,idx4)*deltah(idx3+1,:)*weights(n-1+(idx3-1)*w+idx4,:)';
                 end
             end
             % update weights
             for idx3=1:w
                 for idx4=1:n-1
                     weights(idx4,idx3)=weights(idx4,idx3)-eta*data(idx2,idx4)*deltah(1,idx3);
                 end
                 for idx4=n:n-1+(d-1)*w
                     weights(idx4,idx3)=weights(idx4,idx3)-eta*hid(floor((idx4-n)/w)+1,mod(idx4-n,w)+1)*deltah(ceil((idx4-n)/w)+1,idx3);
                 end
                 for idx4=lw-noLabels+1:lw
                     weights(idx4,idx3)=weights(idx4,idx3)-eta*hid(d,idx3)*deltao(idx4-(lw-noLabels));
                 end
             end
    %         inra(idx2,:)=hid;
         end
         train_error(idx)=error/m;
         error_test=0;
         for idx2=1:m1
             for idx3=1:w
                 sh(1,idx3)=dataTest(idx2,1:n1-1)*weights(1:n1-1,idx3);
                 hid(1,idx3)=1/(1+exp(-sh(1,idx3)));
             end
             for idx3=2:d
                 for idx4=1:w
                     sh(idx3,idx4)=hid(idx3-1,:)*weights(n+(idx3-2)*w:n1-1+(idx3-1)*w,idx4);
                     hid(idx3,idx4) = 1/(1+exp(-sh(idx3,idx4)));
                 end
             end
             for idx3=1:noLabels
                 so(idx3)=hid(d,:)*weights(n1-1+(d-1)*w+idx3,:)';
                 out(idx3)=1/(1+exp(-so(idx3)));
             end
             [ma1, id1]=max(out);
             if abs(id1-dataTest(idx2,n1)-1)>0
                 error_test=error_test+1;
             end
         end
         test_error(idx)=error_test/m1;   
    end
    figure(1); fig1 = plot(train_error);
    figure(2); fig2=plot(test_error);
end