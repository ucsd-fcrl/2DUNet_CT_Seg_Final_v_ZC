% this script can automatically exclude the disconnected parts in the
% segmentation and only keep the largest connectivity.
%%
close all;
clear all;
addpath(genpath('/Users/zhennongchen/Documents/GitHub/Volume_Rendering_by_DL/matlab/'));
%% Find patient list
patient_list = Find_all_folders('/Volumes/Seagate_5T/Ashish_ResyncCT/predicted_seg/');

class_list = []; id_list = [];
for i = 1:size(patient_list,1)
    a = Find_all_folders([patient_list(i).folder,'/',patient_list(i).name]);
    for ii = 1:size(a,1)
        class = split(a(ii).folder,'/');
        class = class(end); class = class{1};
        class_list = [class_list;convertCharsToStrings(class)];
        id_list = [id_list;convertCharsToStrings(a(ii).name)];
    end
   
end
% %%
% class_list = []; id_list = [];
% for i = 1:size(patient_list,1)
%     class = split(patient_list(i).folder,'/');
%     class = class(end); class = class{1};
%     class_list = [class_list;convertCharsToStrings(class)];
%     id_list = [id_list;convertCharsToStrings(patient_list(i).name)];
% end
% patient_list = Find_all_folders('/Volumes/Seagate_5T/2020_CT_data/predicted_seg/Normal/');
% for i = 1:size(patient_list,1)
%     class = split(patient_list(i).folder,'/');
%     class = class(end); class = class{1};
%     class_list = [class_list;convertCharsToStrings(class)];
%     id_list = [id_list;convertCharsToStrings(patient_list(i).name)];
% end
%% Do pixel clearning:
main_folder = '/Volumes/Seagate_5T/Ashish_ResyncCT/predicted_seg/';

% Global_disconnect_cases = [];
% LA_disconnect_cases = [];
% LVOT_disconnect_cases = [];
for i = 1:size(id_list,1)
    patient_class = convertStringsToChars(class_list(i,:));
    patient_id = convertStringsToChars(id_list(i));
    disp(patient_id)
    patient_folder = [main_folder,patient_class,'/',patient_id,'/seg-pred-0.625-4classes-raw/'];
    
    if isfolder(patient_folder) == 1
        
       % check whether it's done:
       if isfile([main_folder,patient_class,'/',patient_id,'/seg-pred-0.625-4classes-connected-mat/pred_s_0.mat']) == 1
           disp(['already done'])
           continue
       end
       
       % make save_folder
       save_folder = [main_folder,patient_class,'/',patient_id,'/seg-pred-0.625-4classes-connected-mat/'];
       mkdir(save_folder)
        
       nii_list = Sort_time_frame(Find_all_files(patient_folder),'_');
       
       Global_disconnect = 0;
       LA_disconnect = 0;
       LVOT_disconnect = 0;
       
       for j = 1: size(nii_list,1)
           file_name = [patient_folder,convertStringsToChars(nii_list(j))];
           data = load_nii(file_name);
           image = data.img;
           
           % exclude
           % First step: Get rid of any disconnected object
           BW = image > 0;
           [BW,image,change] = Find_largest_connected_component_3d(BW,image,0,6);
           if change == 1
               Global_disconnect = 1;
           end
           % check
           CC = bwconncomp(BW,6);
           numPixels = cellfun(@numel,CC.PixelIdxList);
           if size(numPixels,2) > 1
               error('Error occurred in exclusion');
           end
           
           
           % Second step: Turn disconnected object with label > 1 into
           % label = 1
           % (LA = 2, LVOT = 4)
           BW = image == 2;
           [BW,image,change] = Find_largest_connected_component_3d(BW,image,1,26);
           if change == 1
               LA_disconnect = 1;
           end
           BW = image == 4;
           [BW,image,change] = Find_largest_connected_component_3d(BW,image,1,26);
           if change == 1
               LVOT_disconnect = 1;
           end
           
           % Third step: Get rid of disconnected object with label = 1
           BW = image == 1;
           [BW,image,~] = Find_largest_connected_component_3d(BW,image,0,26);
           
           % put back to segmentation
           [image] = Transform_between_nii_and_mat_coordinate(image,1);
           % save
           t = Find_time_frame(convertStringsToChars(nii_list(j)),'_');
           save([save_folder,'/pred_s_',num2str(t),'.mat'],'image')
       end  
        
    else
        disp('Do NOT have seg-pred folder') 
    end          
end

        
    