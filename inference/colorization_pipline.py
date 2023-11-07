import argparse
import cv2
import numpy as np
import os
import time
from tqdm import tqdm
import torch
from basicsr.archs.ddcolor_arch import DDColor
import torch.nn.functional as F
import kornia
from kornia.image import Image, PixelFormat, ChannelsOrder, ImageLayout, ImageSize
import torchvision

class ImageColorizationPipeline(object):

    def __init__(self, model_path, input_size=512):
        
        self.input_size = input_size
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device('cpu')

        self.decoder_type = "MultiScaleColorDecoder"

        if self.decoder_type == 'MultiScaleColorDecoder':
            self.model = DDColor(
                'convnext-l',
                decoder_name='MultiScaleColorDecoder',
                input_size=[self.input_size, self.input_size],
                num_output_channels=2,
                last_norm='Spectral',
                do_normalize=False,
                num_queries=100,
                num_scales=3,
                dec_layers=9,
            ).to(self.device)
        else:
            self.model = DDColor(
                'convnext-l',
                decoder_name='SingleColorDecoder',
                input_size=[self.input_size, self.input_size],
                num_output_channels=2,
                last_norm='Spectral',
                do_normalize=False,
                num_queries=256,
            ).to(self.device)

        self.model.load_state_dict(
            torch.load(model_path, map_location=torch.device('cpu'))['params'],
            strict=False)
        self.model.eval()


    @torch.no_grad()
    def process(self, img):

        size = (720, 1280)
        self.height, self.width = size

        start_time_p1 = time.time()
        img = cv2.resize(img, (1280, 720))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # if self.width * self.height < 100000:
        #     self.input_size = 256

        img = (img / 255.0).astype(np.float32)
        orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]  # (h, w, 1)

        # resize rgb image -> lab -> get grey -> rgb
        img = cv2.resize(img, (self.input_size, self.input_size))
        img_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]
        img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
        img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)

        stop_time_p1 = time.time()

        tensor_gray_rgb = torch.from_numpy(img_gray_rgb.transpose((2, 0, 1))).float().unsqueeze(0).to(self.device)
        output_ab = self.model(tensor_gray_rgb).cpu() # (1, 2, self.height, self.width)

        start_time_p2 = time.time()

        # resize ab -> concat original l -> rgb
        output_ab_resize = F.interpolate(output_ab, size=(self.height, self.width))[0].float()
        output_ab_resize = np.asarray(output_ab_resize).transpose(1, 2, 0)

        output_lab = np.concatenate((orig_l, output_ab_resize), axis=-1)
        output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)

        output_img = (output_bgr * 255.0).round().astype(np.uint8)

        #postprocessing
        output_img = cv2.rotate(output_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        stop_time_p2 = time.time()
        duration_cpu_p1 = stop_time_p1 - start_time_p1
        duration_cpu_p2 = stop_time_p2 - start_time_p2
        duration_cpu_tot = duration_cpu_p1 + duration_cpu_p2
        # print("Duration preprocessing OpenCV: ", duration_cpu_p1, "Duration post-processing OpenCV: ", duration_cpu_p2, "Total duration OpenCV: ", duration_cpu_tot)

        return output_img
    

        # ------ WITH KORNIA ------

        #start_time_p1 = time.time()
        # size = (720, 1280)
        # self.height, self.width = size

        # img_gpu = torch.from_numpy((img / 255.0).astype(np.float32)).to(self.device)     # (img height, img width)
        # img_gpu = img_gpu.unsqueeze(0).unsqueeze(0)            # (1, 1, img height, img width)
        # img_gpu = kornia.geometry.transform.resize(input=img_gpu, size=size)     # (1, 1, self.height, self.width)
        # # img_gpu = kornia.color.grayscale_to_rgb(img_gpu)         # (1, 3, self.height, self.width)
 

        # # NB: BGR to RGB not done because all canals are equal
        # # Keep the original L channel (that we already have in the input) to concatenate with the a,b channels prediction of the model
        # # orig_l_gpu = kornia.color.rgb_to_lab(img_gpu)[:, :1, :, :]           # (1, 1, self.height, self.width)
        # orig_l_gpu = img_gpu * 100          # (1, 1, self.height, self.width)


        # img_gpu = kornia.geometry.transform.resize(input=img_gpu, size=(self.input_size, self.input_size))     # (1, 3, self.input_size, self.input_size) RGB
        # img_l_gpu = img_gpu * 100
        # # img_l_gpu = kornia.color.rgb_to_lab(img_gpu)[:, :1, :, :]             # (1, 1, self.input_size, self.input_size) L channel
        # img_gray_lab_gpu = torch.cat((img_l_gpu, torch.zeros_like(img_l_gpu), torch.zeros_like(img_l_gpu)), dim=1) # (1, 3, self.input_size, self.input_size)
        # img_gray_rgb = kornia.color.lab_to_rgb(img_gray_lab_gpu) # (1, 3, H, W)
        # tensor_gray_rgb = img_gray_rgb.float()

        # #stop_time_p1 = time.time()

        # output_ab = self.model(tensor_gray_rgb) # (1, 2, self.height, self.width)

        #  # Resize ab -> concat original L channel -> rgb -> bgr -> rotate 90° -> permute axis to (Channel, height, width)
        # #start_time_p2 = time.time()

        # #start = time.time()
        # output_ab_resize = kornia.geometry.transform.resize(input=output_ab, size=(self.height, self.width)).float()     # (1, 2, self.height, self.width)
        # #stop = time.time()
        # #print("Resize time", stop - start)

        # #start = time.time()        
        # output_lab = torch.cat((orig_l_gpu, output_ab_resize), dim=1)    # (1, 3, self.height, self.width)
        # #stop = time.time()
        # #print("Cat time", stop - start)

        # #start = time.time() 
        # output_rgb = kornia.color.lab_to_rgb(output_lab)    # (1, 3, self.height, self.width)
        # #stop = time.time()
        # #print("lab to rgb time", stop - start)

        # #start = time.time()
        # output_bgr = kornia.color.rgb_to_bgr(output_rgb)    # (1, 3, self.height, self.width)
        # #stop = time.time()
        # #print("rgb to bgr time", stop - start)

        # output_img = (output_bgr * 255.0).round()
        # # output_img = output_lab


        # rotation_tensor = torch.tensor([90.]).to(self.device)
        # #start = time.time()
        # output_img = kornia.geometry.transform.rotate(tensor=output_img, angle=rotation_tensor)
        # #stop = time.time()
        # #print("Rotation time", stop - start)
        
        # #start = time.time()
        # output_img = np.asarray(output_img.squeeze(0).permute(1, 2, 0).cpu()).astype(np.uint8)
        # #stop = time.time()
        # #print("To numpy + permute time", stop - start)

        # #stop_time_p2 = time.time()

        # #duration_cpu_p1 = stop_time_p1 - start_time_p1
        # #duration_cpu_p2 = stop_time_p2 - start_time_p2
        # #duration_cpu_tot = duration_cpu_p1 + duration_cpu_p2
        # #print("Duration preprocessing Kornia:", duration_cpu_p1, "Duration post-processing Kornia:", duration_cpu_p2, "Total duration Kornia:", duration_cpu_tot)

        # return output_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='pretrain/net_g_200000.pth')
    parser.add_argument('--input', type=str, default='figure/', help='input test image folder or video path')
    parser.add_argument('--output', type=str, default='results', help='output folder or video path')
    parser.add_argument('--input_size', type=int, default=512, help='input size for model')

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    img_list = os.listdir(args.input)
    assert len(img_list) > 0

    colorizer = ImageColorizationPipeline(model_path=args.model_path, input_size=args.input_size)

    for name in tqdm(img_list):
        img = cv2.imread(os.path.join(args.input, name))
        image_out = colorizer.process(img)
        cv2.imwrite(os.path.join(args.output, name), image_out)


if __name__ == '__main__':
    main()
