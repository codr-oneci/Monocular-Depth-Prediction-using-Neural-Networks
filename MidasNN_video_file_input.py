"""Compute depth maps for images in the input folder.
"""
import os
import glob
import torch
import utils
import cv2
import argparse
import matplotlib.pyplot as plt
import time
import numpy as np

from torchvision.transforms import Compose
from midas.dpt_depth import DPTDepthModel
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small
from midas.transforms import Resize, NormalizeImage, PrepareForNet


def run(input_path, output_path, model_path, model_type="dpt_large", optimize=True):
    """Run MonoDepthNN to compute depth maps.
    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """
    print("initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    # load network
    if model_type == "dpt_large": # DPT-Large
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid": #DPT-Hybrid
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode="minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "midas_v21":
        model = MidasNet(model_path, non_negative=True)
        net_w, net_h = 384, 384
        resize_mode="upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    elif model_type == "midas_v21_small":
        model = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
        net_w, net_h = 256, 256
        resize_mode="upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        print(f"model_type '{model_type}' not implemented, use: --model_type large")
        assert False

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


    model.eval()

    if optimize==True:
        # rand_example = torch.rand(1, 3, net_h, net_w)
        # model(rand_example)
        # traced_script_module = torch.jit.trace(model, rand_example)
        # model = traced_script_module

        if device == torch.device("cuda"):
            model = model.to(memory_format=torch.channels_last)
            model = model.half()

    model.to(device)

    # get input
    cap = cv2.VideoCapture(input_path)
    # create output folder
    os.makedirs(output_path, exist_ok=True)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))



    #out = cv2.VideoWriter(output_path+'/outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))


    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #outVidWr = cv2.VideoWriter(output_path+'/outpy.avi', fourcc, 30, (frame_width, frame_height), 0)

    writer = None
    (W, H) = (None, None)


    print("start processing")
    n=0
    m=0
    while(cap.isOpened()) and m<100:
        n+=1

        print("processing frame "+str(n))

        # input

        ret, frame = cap.read()
        if ret == True:
            img_input = transform({"image": frame})["image"]

            # compute
            with torch.no_grad():
                sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
                if optimize==True and device == torch.device("cuda"):
                    sample = sample.to(memory_format=torch.channels_last)
                    sample = sample.half()
                prediction = model.forward(sample)
                #print(type(prediction))
                #print(prediction)
                prediction = (torch.nn.functional.interpolate(prediction.unsqueeze(1),size=frame.shape[:2],mode="bicubic",align_corners=False,).squeeze().cpu().numpy())
                #print(prediction.shape[:])
            # output


            #filename = os.path.join(
                #output_path, os.path.splitext(os.path.basename(str(n)+'.jpg'))[0]
            #)
            #utils.write_depth(filename, prediction, bits=2)
            #prediction = cv2.cvtColor(prediction,cv2.COLOR_GRAY2BGR)


            bits=2

            depth_min = prediction.min()
            depth_max = prediction.max()

            max_val = (2**(8*bits))-1

            if depth_max - depth_min > np.finfo("float").eps:
                out = max_val * (prediction - depth_min) / (depth_max - depth_min)
            else:
                out = np.zeros(prediction.shape, dtype=prediction.type)

            #if bits == 1:
                #cv2.imwrite(path + ".png", out.astype("uint8"))
            #elif bits == 2:
                #cv2.imwrite(path + ".png", out.astype("uint16"))



            scale_percent = 50 # percent of original size
            width = int(out.shape[1] * scale_percent / 100)
            height = int(out.shape[0] * scale_percent / 100)
            dim = (width, height)

            # resize image
            resized = cv2.resize(out, dim, interpolation = cv2.INTER_AREA)




            prediction = resized.astype("uint16")
            prediction=np.repeat(prediction[:, :, np.newaxis], 3, axis=2) #transform 2D numpy array into 3 channel tensor for image representation

            prediction = (prediction * 255.0/np.amax(prediction)).astype("uint8")

            print(prediction.shape[::])
            print(frame.shape[::])
            print(prediction)
            print(frame)


            #outVidWr.write(resized.astype("uint16"))
            if writer is None: #initialize our video writer
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(output_path+'/Infinite_dpt_large.avi', fourcc, 30,(prediction.shape[1], prediction.shape[0]), True)
            # write the output frame to disk

            #cv2.imshow("frame",prediction)
            #cv2.waitKey(50)

            writer.write(prediction)

        else:
            m+=1




    cap.release()
    writer.release()
    #out.release()
    cv2.destroyAllWindows()


    print("finished")



run('input/Infinite.MP4', 'output', "weights/dpt_large-midas-2f21e586.pt", 'dpt_large', optimize=True) #best results with GPU
#run('input/Infinite.MP4', 'output', "weights/midas_v21_small-70d6b9c8.pt", 'midas_v21_small', optimize=True) #embedded systems
#run('input/LAB3.MP4', 'output', "weights/dpt_large-midas-2f21e586.pt", 'dpt_large', optimize=True)
