import yaml
from easydict import EasyDict
import os
import time
from shutil import copyfile
import random
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from network.KAD_Net import *
from utils import *
import numpy as np


def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def main():

    seed_torch(42) # it doesnot work if the mode of F.interpolate is "bilinear"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('/root/autodl-tmp/hsj/KAD_Net/cfg/train_KAD_Net.yaml', 'r') as f:
        args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))

    project_name = args.project_name
    epoch_number = args.epoch_number
    batch_size = args.batch_size
    lr = args.lr
    beta1 = args.beta1
    image_size = args.image_size
    message_length = args.message_length
    message_range = args.message_range
    attention_encoder = args.attention_encoder
    attention_decoder = args.attention_decoder
    weight = args.weight
    dataset_path = args.dataset_path
    save_images_number = args.save_images_number
    noise_layers_R = args.noise_layers.pool_R
    noise_layers_F = args.noise_layers.pool_F

    
    #Set the project and result saving path.
    branch_type = args.branch_type  # "ST" or "FD"
    project_name = f"{branch_type}_KAD_Net_{image_size}_{message_length}_{message_range}_{lr}_{beta1}_{attention_encoder}_{attention_decoder}"


    for i in weight:
        project_name += "_" +  str(i)
    branch_type = args.branch_type  
    base_results_path = "/root/autodl-tmp/hsj/KAD_Net/results"
    branch_folder = os.path.join(base_results_path, branch_type)  # results/ST or results/FD
    image_folder = os.path.join(branch_folder, str(image_size))  
    timestamp = time.strftime(project_name + "_%Y_%m_%d_%H_%M_%S", time.localtime())
    result_folder = os.path.join(image_folder, timestamp) + "/"
    
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)  

    if not os.path.exists(result_folder + "images/"): os.mkdir(result_folder + "images/")
    if not os.path.exists(result_folder + "models/"): os.mkdir(result_folder + "models/")
    copyfile("/root/autodl-tmp/hsj/KAD_Net/cfg/train_KAD_Net.yaml", result_folder + "train_KAD_Net.yaml")
    writer = SummaryWriter('runs/'+ project_name + time.strftime("%_Y_%m_%d__%H_%M_%S", time.localtime()))   

    branch_type = args.branch_type
    network = Network(branch_type, message_length, noise_layers_R, noise_layers_F, device, batch_size, lr, beta1, attention_encoder, attention_decoder, weight)


    train_dataset = attrsImgDataset(os.path.join(dataset_path, "train_" + str(image_size)), image_size, "celebahq")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    print("Training dataset path:", dataset_path)
    print("Number of samples in training dataset:", len(train_dataset))

    val_dataset = attrsImgDataset(os.path.join(dataset_path, "val_" + str(image_size)), image_size, "celebahq")
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print("\nStart training : \n\n")

    for epoch in range(1, epoch_number + 1):

        running_result = {
            "g_loss": 0.0,
            "error_rate_C": 0.0,
            "error_rate_R": 0.0,
            "error_rate_F": 0.0,
            "psnr": 0.0,
            "ssim": 0.0,
            "g_loss_on_discriminator": 0.0,
            "g_loss_on_encoder_MSE": 0.0,
            "g_loss_on_encoder_LPIPS": 0.0,
            "g_loss_on_decoder_C": 0.0,
            "g_loss_on_decoder_R": 0.0,
            "g_loss_on_decoder_F": 0.0,
            "d_loss": 0.0
        }

        start_time = time.time()

        '''
        train
        '''
        for step, (image, mask) in enumerate(train_dataloader, 1):

            print(device)
            image = image.to(device)
            message = torch.Tensor(np.random.choice([-message_range, message_range], (image.shape[0], message_length))).to(device)

            if args.branch_type == "ST":
                result = network.train_ST(image, message, mask)
            elif args.branch_type == "FD":
                result = network.train_FD(image, message, mask)
            else:
                raise ValueError("Invalid branch_type. Use 'ST' or 'FD'.")

            print('Epoch: {}/{} Step: {}/{}'.format(epoch, epoch_number, step, len(train_dataloader)))
           
            for key in result:
                if result[key] is not None:  
                    print(key, float(result[key]))
                    writer.add_scalar("Train/" + key, float(result[key]), (epoch - 1) * len(train_dataloader) + step)
                    running_result[key] += float(result[key])
                else:
                    print(key, "Skipped (None Value)")


        '''
        train results
        '''
        content = "Epoch " + str(epoch) + " : " + str(int(time.time() - start_time)) + "\n"
        for key in running_result:
            content += key + "=" + str(running_result[key] / step) + ","
            writer.add_scalar("Train_epoch/" + key, float(running_result[key] / step), epoch)
        content += "\n"

        with open(result_folder + "/train_log.txt", "a") as file:
            file.write(content)
        print(content)

        '''
        validation
        '''

        val_result = {
            "g_loss": 0.0,
            "error_rate_C": 0.0,
            "error_rate_R": 0.0,
            "error_rate_F": 0.0,
            "psnr": 0.0,
            "ssim": 0.0,
            "g_loss_on_discriminator": 0.0,
            "g_loss_on_encoder_MSE": 0.0,
            "g_loss_on_encoder_LPIPS": 0.0,
            "g_loss_on_decoder_C": 0.0,
            "g_loss_on_decoder_R": 0.0,
            "g_loss_on_decoder_F": 0.0,
            "d_loss": 0.0
        }

        start_time = time.time()

        saved_iterations = np.random.choice(np.arange(1, len(val_dataloader)+1), size=save_images_number, replace=False)
        saved_all = None

        for step, (image, mask) in enumerate(val_dataloader, 1):
            image = image.to(device)
            message = torch.Tensor(np.random.choice([-message_range, message_range], (image.shape[0], message_length))).to(device)

            if args.branch_type == "ST":
                result, (images, encoded_images, noised_images) = network.validation_ST(image, message, mask)
            elif args.branch_type == "FD":
                result, (images, encoded_images, noised_images) = network.validation_FD(image, message, mask)
            else:
                raise ValueError("Invalid branch_type. Use 'ST' or 'FD'.")

            print('Epoch: {}/{} Step: {}/{}'.format(epoch, epoch_number, step, len(val_dataloader)))
            for key in val_result:
                if result[key] is not None: 
                    print(key, float(result[key]))
                    writer.add_scalar("Val/" + key, float(result[key]), (epoch - 1) * len(val_dataloader) + step)
                    val_result[key] += float(result[key])
                else:
                    print(key, "Skipped (None Value)")


            if step in saved_iterations:
                if saved_all is None:
                    saved_all = get_random_images(image, encoded_images, noised_images)
                else:
                    saved_all = concatenate_images(saved_all, image, encoded_images, noised_images)

        save_images(saved_all, epoch, result_folder + "images/", resize_to=None)

        '''
        validation results
        '''
        content = "Epoch " + str(epoch) + " : " + str(int(time.time() - start_time)) + "\n"
        for key in val_result:
            content += key + "=" + str(val_result[key] / step) + ","
            writer.add_scalar("Val_epoch/" + key, float(val_result[key] / step), epoch)
        content += "\n"

        with open(result_folder + "/val_log.txt", "a") as file:
            file.write(content)
        print(content)

        '''
        save model
        '''
        path_model = result_folder + "models/"
        path_encoder_decoder = path_model + "EC_" + str(epoch) + ".pth"
        path_discriminator = path_model + "D_" + str(epoch) + ".pth"
        network.save_model(path_encoder_decoder, path_discriminator)

        writer.close()


if __name__ == '__main__':
    main()
