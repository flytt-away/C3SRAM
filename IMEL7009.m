%load data
data=importdata('bMAC.csv');

pre_data = data.data(:,2);
post_data = data.data(:,4);
% plot(pre_data)
% hold on
% plot(post_data)
% legend('Pre - Data', 'Post - Data');

pre_rise_data = pre_data(1:65);
pre_fall_data = pre_data(66:130);
pre_temp = pre_fall_data;
for i=1:65
   pre_fall_data(i) = pre_temp(66-i); 
end

post_rise_data = post_data(1:65);
post_fall_data = post_data(66:130);
post_temp = post_fall_data;
for i=1:65
   post_fall_data(i) = post_temp(66-i); 
end

%data process
post_process_data = (post_rise_data + post_fall_data)/2;
pre_process_data = (pre_rise_data + pre_fall_data)/2;

bmac_value = -64:2:64;
bmac_value = bmac_value.';

ideal_data = (0.8-0)/(64-(-64)).*bmac_value + 0.4;

%plot(bmac_value,pre_process_data)
%hold on
%plot(bmac_value,post_process_data)
%hold on
%plot(bmac_value,ideal_data)
%legend('Pre - Data', 'Post - Data', 'Ideal - Data');
%fit curve
post_polyfit = polyfit(bmac_value,post_process_data,1);
post_slope = post_polyfit(1);
post_bias = post_polyfit(2);
disp("slope")
disp(post_slope)
disp("bias")
disp(post_bias)

post_fit = post_slope*bmac_value+post_bias;



%RMSE
n = length(x);

residuals = ideal_data - post_fit;

RMSE = sqrt(sum((ideal_data - post_process_data).^2)/n);
disp("RMSE")
disp(RMSE)

epsilon = zeros(size(bmac_value));
for i=1:n
    epsilon(i) = RMSE*randn();
end

figure()
plot(bmac_value,post_process_data,'o','MarkerSize', 5)
hold on 
plot(bmac_value,post_fit,'-','LineWidth', 2)
hold on 

plot(bmac_value,ideal_data,'-','LineWidth', 2)
hold on 
xlim([-64, 64]);
xlabel('bMAC Value'); % 设置 x 轴标题
ylabel('MBL Voltage'); % 设置 y 轴标题
%y_fit = ideal_data + epsilon;
%plot(bmac_value,y_fit,'-');
legend('Post', 'Post-fit','Ideal', 'LineWidth', 1);