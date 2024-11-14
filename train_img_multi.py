import numpy as np
import os
import configargparse
import matplotlib.pyplot as plt

p = configargparse.ArgumentParser()
p.add_argument('--final_path', type=str, default='./logs', help='Root for load npy file')
# p.add_argument('--label', type=str, default='Pre-training epochs', help='Label for the image')
# p.add_argument('--label_parameter', type=str, default= "0 100 200 300 400 3000", help='List of label parameters separated by space')
# p.add_argument('--file_label', type=str, default='epochs=', help='Label for the target file')
opt = p.parse_args()

#label = ['Baseline','mixup','Dynamic-mixup']
# label = ['Baseline','CutMix','Dynamic-CutMix']
#label = ['Baseline','Dynamic-noise mixup','Dynamic-noise CutMix']

#motivition label
# label = ['Baseline','Blank mixup','Simple mixup','Complex mixup','Noise mixup']
# label = ['Baseline','Blank CutMix','Simple CutMix','Complex CutMix','Noise CutMix']
# label = ['Baseline','Noise CUtMix','Noise CUtMix']
# label = ['Baseline','Dynamic-mixup','Dynamic-CutMix','mix-SIREN (mixup)','mix-SIREN (CutMix)']
# label = ['Non','Simple','Complex','Noise']
label = ['SIREN','Mix-SIREN-N-DB','Mix-SIREN-N-DR']
# label = ['SIREN','Mix-SIREN-N-DR']
def plot_and_save_metrics(results_list, txt_save_path,plot_x):
    
    results = [result_list[-3000:] for result_list in results_list]
    min_value = float('inf')
    max_value = float('-inf')
    
    plt.figure(figsize=(10, 8))
    for i, result_list in enumerate(results):
        epochs = np.arange(1, len(result_list) + 1)  # Epoch 
        psnr_values = [item[0] for item in result_list]
        min_value = min(min_value, min(psnr_values))
        max_value = max(max_value, max(psnr_values))
        
        plt.plot(epochs, psnr_values, label=f'{label[i]}')
    #plt.axvline(x=plot_x, color='red', linestyle='--')  # 添加垂直虚线
    # plt.text(328, 1, r'$T_{\mathrm{R}}$', fontsize=12, color='red', rotation=90)  # 在x=328处添加文字
    # plt.annotate("TR", xy=(320, 0), xytext=(320, -10000), fontsize=12,
    #          arrowprops=dict(facecolor='black', arrowstyle='->'), ha='center')
    plt.xlabel('Epochs', fontsize=22)
    plt.ylabel('PSNR', fontsize=22)
    # plt.title('PSNR Over Pre-training Epochs')
    plt.legend(fontsize=22,loc='upper left')
    # plt.legend(fontsize=22)
    plt.grid(True)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlim([1, 3000])  
    plt.ylim([min_value, max_value+1])  
    image_file = 'PSNR.png'
    full_image_path = os.path.join(txt_save_path, image_file)
    plt.savefig(full_image_path)
    plt.close()
    # import pdb;pdb.set_trace()
    min_value = float('inf')
    max_value = float('-inf')
    plt.figure(figsize=(10, 8))
    for i, result_list in enumerate(results):
        epochs = np.arange(1, len(result_list) + 1)  # Epoch 
        ssim_values = [item[1] for item in result_list] 

        min_value = min(min_value, min(ssim_values))
        max_value = max(max_value, max(ssim_values))
        
        plt.plot(epochs, ssim_values, label=f'{label[i]}')

    #plt.axvline(x=plot_x, color='red', linestyle='--')  # 添加垂直虚线
    plt.xlabel('Epochs', fontsize=22)
    plt.ylabel('SSIM', fontsize=22)
    # plt.title('Metrics Over Pretraining Epochs')
    plt.legend(fontsize=22,loc='upper left')
    plt.grid(True)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlim([1, 3000])  
    plt.ylim([min_value, 1])  
    image_file = 'SSIM.png'
    full_image_path = os.path.join(txt_save_path, image_file)
    plt.savefig(full_image_path)
    plt.close()
    
    min_value = float('inf')
    max_value = float('-inf')
    plt.figure(figsize=(10, 8))
    for i, result_list in enumerate(results):
        epochs = np.arange(1, len(result_list) + 1)  # Epoch 
        l1_values = [item[2] for item in result_list]  

        min_value = min(min_value, min(l1_values))
        max_value = max(max_value, max(l1_values))
        
        plt.plot(epochs, l1_values, label=f'{label[i]}')
    #plt.axvline(x=plot_x, color='red', linestyle='--')  # 添加垂直虚线
    plt.xlabel('Epochs', fontsize=22)
    plt.ylabel('L1', fontsize=22)
    plt.legend(fontsize=22,loc='upper left')
    # plt.title('Metrics Over Pretraining Epochs')
    plt.legend(fontsize=22)
    plt.grid(True)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlim([1, 3000])  
    plt.ylim([min_value, max_value])  
    image_file = 'L1.png'
    full_image_path = os.path.join(txt_save_path, image_file)
    plt.savefig(full_image_path)
    plt.close()
    

    print(f"Metrics graph saved to: {full_image_path}")

    
pretrain_folder = opt.final_path
# label_parameter = opt.label_parameter.split()
# label_parameter = [int(x) for x in label_parameter]
# pretraining_epochs = [0, 100, 200, 300, 400, 3000]
label_parameter = label
results_lists = []

pretrain_folder_name = sorted(os.listdir(pretrain_folder))
i = 0
for epoch_folder in pretrain_folder_name:
    # import pdb;pdb.set_trace()
    if not os.path.isdir(os.path.join(pretrain_folder, epoch_folder)):
        continue
    file_path = os.path.join(pretrain_folder, epoch_folder,'checkpoints', "result_list_average.npy")
    print(file_path)
    # import pdb;pdb.set_trace()
    if os.path.exists(file_path):
        result_list = np.load(file_path, allow_pickle=True)
        results_lists.append(result_list)
    i = i + 1 
# pretrain_folder = os.path.join(pretrain_folder, epoch_folder)
# import pdb;pdb.set_trace()
# 调用绘图函数os.path.join(pretrain_folder, epoch_folder,
plot_and_save_metrics(results_lists, pretrain_folder,plot_x= 230)