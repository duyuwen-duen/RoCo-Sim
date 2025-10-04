"""
This file not related to blender python API
"""
import numpy as np
import imageio.v2 as imageio
import glob
import os
import sys
import cv2
import OpenImageIO as oiio
import imageio.v3 as iio

def exr_to_png(exr_path, output_path, occlusion_mask=None):
    """read EXR and save as PNG"""
    inp = oiio.ImageInput.open(exr_path)
    if not inp:
        print(f"❌ Failed to open EXR: {exr_path}")
        return
    spec = inp.spec()
    data = inp.read_image(format=oiio.BASETYPE.FLOAT)
    inp.close()
    image = np.asarray(data).reshape(spec.height, spec.width, spec.nchannels)
    image = np.clip(image, 0.0, 1.0)
    image_u8 = (image * 255).astype(np.uint8)
    image_bin = np.where(image_u8 > 150, 255, 0).astype(np.uint8)

    if occlusion_mask is not None:
        if occlusion_mask.ndim == 2 and image_bin.ndim == 3:
            occlusion_mask = occlusion_mask[..., None]
        masked_output = image_bin * occlusion_mask
    else:
        masked_output = image_bin

    iio.imwrite(output_path, masked_output)
    print(f"✅ Image saved to {output_path}")

 
def motion_blur(image, degree=12, angle=45):
    """
    degree : intensity of blur
    angle : direction of blur. 
        angle = 0 is +u, angle = 90 is -v
    """
    image = np.array(image)
    angle -= 135  # because np.diag create a 135 degree rotated kernel
 
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
 
    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred
 

def read_from_render(rendered_output_dir,
                     image_type, 
                     image_prefix):
    """
    Args:
        image_type: str,
            RGB/depth/mask
        image_prefix: str,
            vehicle_only/vehicle_and_plane/plane_only
    """

    files = glob.glob(os.path.join(rendered_output_dir, image_type) + f"/{image_prefix}*")
    assert len(files) == 1
    file_path = files[0]
    if ".exr" in file_path:
        return read_exr(file_path)
    else:
        return imageio.imread(file_path)

def read_exr(path):
    """use OpenImageIO to read EXR file"""
    inp = oiio.ImageInput.open(path)
    if not inp:
        raise RuntimeError(f"Failed to open EXR: {path}")
    spec = inp.spec()
    data = inp.read_image(format=oiio.BASETYPE.FLOAT)
    inp.close()

    arr = np.asarray(data).reshape(spec.height, spec.width, spec.nchannels)
    return arr


def expand_depth(depth_vp, neighbor_size=1, bg_value=1e10):
    """
        depth_vp: np.ndarray
            shape: [H, W, 1]
        neighbor_size: int
            kernel size, but not used currently
    """
    bg_pos = depth_vp == bg_value
    neighbor_depth = np.zeros((depth_vp.shape[0], depth_vp.shape[1], 5))
    neighbor_depth[:,:,0] = depth_vp[:,:,0]
    
    for idx, axis_shift in enumerate([(0,1),(0,-1),(1,1),(1,-1)]):
        axis, shift = axis_shift
        depth_vp_new = np.roll(depth_vp[:,:,0], axis=axis, shift=shift)
        if axis == 0:
            target = 0 if shift == 1 else -1
            depth_vp_new[target] = bg_value

        if axis == 1:
            target = 0 if shift == 1 else -1
            depth_vp_new[:,target] = bg_value

        neighbor_depth[:,:,idx + 1] = depth_vp_new

    bg_neighbor_depth = np.min(neighbor_depth, axis=2)[...,np.newaxis]

    return bg_pos * bg_neighbor_depth + (1 - bg_pos) * depth_vp




def compose(rendered_opt,
            background_RGB,
            background_depth,
            render_downsample,
            motion_blur_degree,
            depth_and_occlusion,
            save_path,
            camera_name):
    """
    Args:
        rendered_output_dir: str
            the folder that saves RGB/depth/mask for vehicle w/wo plane
        background_RGB: np.ndarray
            background image
        background_depth: np.ndarray
            background depth image
    """
    rendered_output_dir = os.path.join(rendered_opt['output_dir'],rendered_opt['setting_type'],'cache',rendered_opt['road_seq'],rendered_opt['seq'])
    RGB_over_background = read_from_render(os.path.join(rendered_output_dir, camera_name), 'RGB', 'vehicle_and_shadow_over_background') # [H, W, 4]
    
    if depth_and_occlusion == False:
        H, W = RGB_over_background.shape[:2]
        RGB_over_background = cv2.resize(RGB_over_background, (render_downsample * W, render_downsample * H))
        RGB_over_background = RGB_over_background[:,:,:3]
        # RGB_over_background = motion_blur(RGB_over_background, degree=motion_blur_degree, angle = 45)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        imageio.imsave(os.path.join(save_path,rendered_opt['render_name']+'.jpg'), RGB_over_background)

        return

    depth_vp = read_from_render(os.path.join(rendered_output_dir, camera_name), 'depth', 'vehicle_and_plane') # [H, W, 4]
    depth_vp = expand_depth(depth_vp).astype(np.float32)
    mask_vs = read_from_render(os.path.join(rendered_output_dir, camera_name),'mask', 'vehicle_and_shadow') # [H, W, 4]
    H, W = depth_vp.shape[:2]
    # upsample
    depth_vp = cv2.resize(depth_vp, (render_downsample * W, render_downsample * H))
    RGB_over_background = cv2.resize(RGB_over_background, (render_downsample * W, render_downsample * H))
    RGB_over_background = RGB_over_background[:,:,:3]
    # RGB_over_background = motion_blur(RGB_over_background, degree=motion_blur_degree, angle = 45)
    mask_vs = cv2.resize(mask_vs, (render_downsample * W, render_downsample * H))
    depth_vp = depth_vp[:,:,0:1] # one channel is enough. have no alpha channel

    # mask_vs [H, W, 3]
    # The 3 channels should be all the same
    mask_vs = mask_vs[:,:,0:1] # range [0, 255]

    noise_thres = 0.02
    # we only care about depth of vehicle and shadow, not plane.
    # except for the vehicle + shadow, we set the depth to inf. (use background)
    depth_vs = np.where(mask_vs > noise_thres, depth_vp, np.inf)
    # still aliasing.
    background_depth = np.expand_dims(background_depth[:,:,0], axis=-1)
    occlusion_mask = (depth_vs <= background_depth).astype(np.uint8)
    print("background_depth", background_depth.shape)
    RGB_composite = np.where(depth_vs <= background_depth, 
                             RGB_over_background, background_RGB ).astype(np.uint8)
    
    imageio.imsave(os.path.join(rendered_output_dir,camera_name, "RGB_composite.png"), RGB_composite)
    imageio.imsave(os.path.join(rendered_output_dir,camera_name, "RGB_over_background.png"), RGB_over_background)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # if not os.path.exists(f"{save_path}_cover"):
    #     os.makedirs(f"{save_path}_cover")
    # imageio.imsave(os.path.join(f"{save_path}_cover",rendered_opt['render_name']+'.jpg'), RGB_over_background)
    imageio.imsave(os.path.join(save_path,rendered_opt['render_name']+'.jpg'), RGB_composite)
    print("save to", os.path.join(save_path,rendered_opt['render_name']+'.jpg'))
    exr_path = os.path.join(rendered_output_dir,camera_name, "mask","vehicle_and_shadow0001.exr")
    data_type = save_path.split("/")[-2]
    cam_name = save_path.split("/")[-3]
    mask_save_path = os.path.join(f"{rendered_opt['output_dir']}/{rendered_opt['setting_type']}",f"{rendered_opt['road_seq']}_mask",cam_name,data_type,"image")
    if not os.path.exists(mask_save_path):
        os.makedirs(mask_save_path)
    output_path = os.path.join(mask_save_path,rendered_opt['render_name']+'.png')
    exr_to_png(exr_path, output_path,occlusion_mask)



if __name__ == "__main__":
    rendered_output_dir = sys.argv[1]
    background_RGB = imageio.imread(sys.argv[2])
    H, W, _ = background_RGB.shape
    background_depth = np.full((H,W), 1e8)
    
    compose(rendered_output_dir, background_RGB, background_depth, render_downsample=2, motion_blur=4)