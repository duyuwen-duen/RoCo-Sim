import argparse
import ast
import torch
from PIL import Image
import cv2
import os
import sys
from mobilesamv2.promt_mobilesamv2 import ObjectAwareModel
from mobilesamv2 import sam_model_registry, SamPredictor
from typing import Any, Dict, Generator,List
import matplotlib.pyplot as plt
import numpy as np,json
from tqdm import tqdm

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)

def get_points(label,K,l2c,H,W):
    input_coords = []
    input_labels = []
    for obj in label:
        if obj["type"] == "traffic_signs" or obj["type"] == "construction":
            continue
        loc = np.array([obj['3d_location']['x'], obj['3d_location']['y'], obj['3d_location']['z'],1])
        loc_3d = l2c @ loc.reshape(4,1)
        loc_3d = loc_3d[:3] / loc_3d[3]
        loc_2d = K @ loc_3d
        loc_2d = loc_2d[:2] / loc_2d[2]
        if loc_2d[0] < 0 or loc_2d[0] > W or loc_2d[1] < 0 or loc_2d[1] > H:
            continue
        input_coords.append(loc_2d[:2].squeeze())
        input_labels.append(1)
    input_coords = np.array(input_coords)
    input_labels = np.array(input_labels)
    return input_coords, input_labels

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ObjectAwareModel_path", type=str, default='./PromptGuidedDecoder/ObjectAwareModel.pt', help="ObjectAwareModel path")
    parser.add_argument("--Prompt_guided_Mask_Decoder_path", type=str, default='./PromptGuidedDecoder/Prompt_guided_Mask_Decoder.pt', help="Prompt_guided_Mask_Decoder path")
    parser.add_argument("--encoder_path", type=str, default="./", help="select your own path")
    parser.add_argument("--img_path", type=str, default="./test_images/", help="path to image file")
    parser.add_argument("--imgsz", type=int, default=1024, help="image size")
    parser.add_argument("--iou",type=float,default=0.9,help="yolo iou")
    parser.add_argument("--conf", type=float, default=0.4, help="yolo object confidence threshold")
    parser.add_argument("--retina",type=bool,default=True,help="draw segmentation masks",)
    parser.add_argument("--output_dir", type=str, default="./output", help="image save path")
    parser.add_argument("--encoder_type", choices=['tiny_vit','sam_vit_h','mobile_sam','efficientvit_l2','efficientvit_l1','efficientvit_l0'], help="choose the model type")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument("--calib_path", type=str, default="/home/ubuntu/duyuwen/ChatSim_v2x/data/rcooper/calib/lidar2cam/136.json", help="path to intrinsic file")
    parser.add_argument("--cam_name", type=str, default='136-0', help="camera id")
    parser.add_argument("--label_path", type=str, default="./label/", help="path to label file")
    parser.add_argument("--data_range", type=str, default="-1,-1", help="data range")
    return parser.parse_args()

def create_model():
    Prompt_guided_path='./PromptGuidedDecoder/Prompt_guided_Mask_Decoder.pt'
    obj_model_path='./weight/ObjectAwareModel.pt'
    ObjAwareModel = ObjectAwareModel(obj_model_path)
    PromptGuidedDecoder=sam_model_registry['PromptGuidedDecoder'](Prompt_guided_path)
    mobilesamv2 = sam_model_registry['vit_h']()
    mobilesamv2.prompt_encoder=PromptGuidedDecoder['PromtEncoder']
    mobilesamv2.mask_decoder=PromptGuidedDecoder['MaskDecoder']
    return mobilesamv2,ObjAwareModel
    
def show_anns(anns):
    if len(anns) == 0:
        return
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones((anns.shape[1], anns.shape[2], 4))
    img[:,:,3] = 0
    for ann in range(anns.shape[0]):
        m = anns[ann].bool()
        m=m.cpu().numpy()
        color_mask = np.concatenate([np.random.random(3), [1]])
        img[m] = color_mask
    ax.imshow(img)

def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    assert len(args) > 0 and all(
        len(a) == len(args[0]) for a in args
    ), "Batched iteration must have inputs of all the same size."
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]

encoder_path={'efficientvit_l2':'./weight/l2.pt',
            'tiny_vit':'./weight/mobile_sam.pt',
            'sam_vit_h':'./weight/sam_vit_h.pt',}

def main(args):
    output_dir=args.output_dir  
    mobilesamv2, ObjAwareModel=create_model()
    image_encoder=sam_model_registry[args.encoder_type](encoder_path[args.encoder_type])
    mobilesamv2.image_encoder=image_encoder
    torch.cuda.set_device(args.gpu_id)
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    mobilesamv2.to(device=device)
    mobilesamv2.eval()
    predictor = SamPredictor(mobilesamv2)
    image_files= sorted(os.listdir(args.img_path))
    label_path = args.label_path
    label_list = sorted(os.listdir(label_path))

    camera_intrinsic_path = os.path.join(args.calib_path,f'camera_intrinsic/{args.cam_name}.json')
    with open(camera_intrinsic_path, 'r') as f:
        camera_info = json.load(f)
    camera_intrinsic = np.array(camera_info['intrinsic'])
    
    extrinsic_path = os.path.join(args.calib_path,f'lidar2cam/{args.cam_name}.json')
    with open(extrinsic_path, 'r') as f:
        camera_info = json.load(f)
    l2c = np.array(camera_info['lidar2cam'])
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    begin = int(args.data_range.split(',')[0])
    end = int(args.data_range.split(',')[1])
    if begin != -1 and end != -1:
        image_files = image_files[begin:end]
        label_list = label_list[begin:end]
    elif begin != -1:
        image_files = image_files[begin:]
        label_list = label_list[begin:]
    elif end != -1:
        image_files = image_files[:end]
        label_list = label_list[:end]
        
    idx = 0
    for image_name in tqdm(image_files):
        image = cv2.imread(os.path.join(args.img_path, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H,W,_ = image.shape
        obj_results = ObjAwareModel(image,device=device,retina_masks=args.retina,imgsz=args.imgsz,conf=args.conf,iou=args.iou)
        predictor.set_image(image)
        input_boxes1 = obj_results[0].boxes.xyxy
        input_boxes = input_boxes1.cpu().numpy()
        input_boxes = predictor.transform.apply_boxes(input_boxes, predictor.original_size)
        input_boxes = torch.from_numpy(input_boxes).cuda()
        sam_mask=[]
        image_embedding=predictor.features
        image_embedding=torch.repeat_interleave(image_embedding, 320, dim=0)
        prompt_embedding=mobilesamv2.prompt_encoder.get_dense_pe()
        prompt_embedding=torch.repeat_interleave(prompt_embedding, 320, dim=0)
        for (boxes,) in batch_iterator(320, input_boxes):
            with torch.no_grad():
                image_embedding=image_embedding[0:boxes.shape[0],:,:,:]
                prompt_embedding=prompt_embedding[0:boxes.shape[0],:,:,:]
                sparse_embeddings, dense_embeddings = mobilesamv2.prompt_encoder(
                    points=None,
                    boxes=boxes,
                    masks=None,)
                low_res_masks, _ = mobilesamv2.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=prompt_embedding,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    simple_type=True,
                )
                low_res_masks=predictor.model.postprocess_masks(low_res_masks, predictor.input_size, predictor.original_size)
                sam_mask_pre = (low_res_masks > mobilesamv2.mask_threshold)*1.0
                sam_mask.append(sam_mask_pre.squeeze(1))
        sam_mask=torch.cat(sam_mask)
        
        annotation = sam_mask
        areas = torch.sum(annotation, dim=(1, 2))
        sorted_indices = torch.argsort(areas, descending=True)
        show_img = annotation[sorted_indices]

        background = np.ones_like(image) * 255
        label_file = os.path.join(label_path, label_list[idx])
        with open(label_file, 'r') as f:
            label = json.load(f)
        input_coords, input_labels = get_points(label,camera_intrinsic,l2c,H,W)
       
        for mask in show_img:
            mask_img = np.zeros_like(image)
            
            mask_img[mask.cpu().numpy().astype(bool)] = image[mask.cpu().numpy().astype(bool)]
            area = torch.sum(mask)
            if area > W*H/4:
                continue
            point_in_mask = False
            
            for i in range(input_coords.shape[0]):
                x, y = input_coords[i]
                if mask[int(y), int(x)]:
                    point_in_mask = True
                    break
            if point_in_mask:
                background = np.where(mask_img == 0, background, mask_img)
        background = np.where(background == 255, 0,255)
        mask_image = Image.fromarray((background).astype(np.uint8)[:,:,0])
        mask_image.save(os.path.join(args.output_dir, image_name.replace('.jpg', '.png')))
        idx += 1
        

if __name__ == "__main__":
    args = parse_args()
    main(args)
