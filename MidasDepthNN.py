# import the necessary packages
import os
import glob
import torch
import utils
import cv2
import argparse
import matplotlib.pyplot as plt


from torchvision.transforms import Compose
from midas.dpt_depth import DPTDepthModel
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small
from midas.transforms import Resize, NormalizeImage, PrepareForNet



# determine the device type
print(torch.__version__)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# initialize the midas model using torch hub
modelType = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", modelType)
# flash the model to the device and set it to eval mode
midas.to(DEVICE)
midas.eval()

net_w, net_h = 384, 384
resize_mode = "minimal"
normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


transform = Compose(
    [
        Resize(
            net_w,
            net_h,
            resize_target=None,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method=resize_mode,
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        normalization,
        PrepareForNet(),
    ]
)




#prepare input data for the MidasNN
img_name="5.jpg"
#img = utils.read_image(img_name)
img = cv2.imread(img_name)
if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

img_input = transform({"image": img})["image"]

#End data input section



# turn off auto grad
with torch.no_grad():
	# get predictions from input
	prediction = midas(img_input)
	# unsqueeze the predictions batchwise
	prediction = torch.nn.functional.interpolate(
		prediction.unsqueeze(1), size=[384,384], mode="bicubic",
		align_corners=False).squeeze()
# store the predictions in a numpy array
output = prediction.cpu().numpy()
