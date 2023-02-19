import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time
import torch
import pytorch3d
import imageio
import argparse as ap

from tqdm import trange
from pytorch3d.vis.plotly_vis import plot_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras, 
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)

b = 160 #baseline, mm
f = 3740 #focal length, pixels

#Note images already rectified
def get_disparity(left_im, right_im, wsize, min_disp=10):
    print("Processing {}...".format(data_dir.split("/")[-1]))
    print("Window Size: {}".format(wsize))
    t0 = time.time()
    H = left_im.shape[0]
    W = left_im.shape[1]

    padded_left_im = np.pad(left_im, ((wsize//2, wsize//2), (wsize//2, wsize//2), (0, 0)))
    padded_right_im = np.pad(right_im, ((wsize//2, wsize//2), (wsize//2, wsize//2), (0, 0)))
    match_locs = np.zeros((H*W, 2, 2))
    
    pixel_counter = 0
    for y in trange(H):    
        for x_l in range(W):
            window_left = padded_left_im[y: y + wsize, x_l: x_l + wsize]
            curr_loc = [y, x_l]
            windows = []
            for x_r in range(min(W, x_l + min_disp)):
                window_right = padded_right_im[y: y + wsize, x_r: x_r + wsize]
                windows.append(window_right.reshape(-1, wsize*wsize*3))

            windows = np.concatenate(windows).reshape(-1, wsize*wsize*3)
            min_x = np.argmin(np.sum(np.square(window_left.reshape(-1, wsize*wsize*3) - windows), axis = 1), axis = 0)
            min_loc = [y, min_x]
            match_locs[pixel_counter, :, 0] = curr_loc
            match_locs[pixel_counter, :, 1] = min_loc
            pixel_counter += 1

    left_disp = np.zeros_like(left_im[:, :, 0])
    right_disp = np.zeros_like(right_im[:, :, 0])
    disparities = []
    for i in range(len(match_locs)):
        left_loc = match_locs[i, :, 0][::-1].astype(int)
        right_loc = match_locs[i, :, 1][::-1].astype(int)
        d = np.max([0, left_loc[0] - right_loc[0]])
        disparities.append(d)
        left_disp[left_loc[1], left_loc[0]] = d
        right_disp[right_loc[1], right_loc[0]] = d

    print("Took {} minutes".format((time.time() - t0)/60.0))
    return match_locs, disparities, left_disp, right_disp

def get_depth_from_disparity(matches, disparities, left_im):
    vertices = np.zeros((left_im.shape[0]*left_im.shape[1], 3))
    colors = np.zeros_like(vertices)

    for i, d in enumerate(disparities):
        y, x = matches[i, 0, 0], matches[i, 1, 0]
        depth = (b*f)/(3*(d + 200))
        vertices[i, :] = [x, -y, -depth]
        colors[i] = left_im[int(y), int(x)]

    vertices -= np.mean(vertices, axis=0)
    pcd = pytorch3d.structures.Pointclouds(
        points=torch.tensor(vertices, dtype=torch.float32).unsqueeze(0), 
        features=torch.tensor(colors/255.0, dtype=torch.float32).unsqueeze(0)
    )
    
    return pcd

def visualize_matching(im1, im2, wsize):
    def callback(event):
        new_im1 = im1.copy()
        new_im2 = im2.copy()

        padded_im1 = np.pad(new_im1, ((wsize//2, wsize//2), (wsize//2, wsize//2), (0, 0)))
        padded_im2 = np.pad(new_im2, ((wsize//2, wsize//2), (wsize//2, wsize//2), (0, 0)))

        x, y = [int(event.xdata), int(event.ydata)]
        print('Picked Point: ({}, {})'.format(x, y))
        cv2.line(new_im1, (0, y), (im1.shape[1], y), (0, 0, 255), 1)
        cv2.line(new_im2, (0, y), (im1.shape[1], y), (0, 0, 255), 1)
        cv2.rectangle(new_im1, (x - wsize//2, y - wsize//2), (x + wsize//2, y + wsize//2), (255, 0, 0), 1)
        cv2.circle(new_im1, (x, y), 2, (255, 0, 0), -1)

        window_left = padded_im1[y: y + wsize, x: x + wsize]
        windows = []
        for x_r in range(im1.shape[1]):
            window_right = padded_im2[y: y + wsize, x_r: x_r + wsize]
            windows.append(window_right.reshape(-1, wsize*wsize*3))

        windows = np.concatenate(windows).reshape(-1, wsize*wsize*3)
        matching_costs = np.sum(np.square(window_left.reshape(-1, wsize*wsize*3) - windows), axis = 1)
        matching_costs = ((matching_costs - matching_costs.min())*im1.shape[0])/(matching_costs.max() - matching_costs.min()) 

        min_x = np.argmin(matching_costs, axis=0)

        cv2.rectangle(new_im2, (min_x - wsize//2, y - wsize//2), (min_x + wsize//2, y + wsize//2), (255, 0, 0), 1)
        cv2.line(new_im2, (min_x, 0), (min_x, im1.shape[0]), (255, 0, 0), 1)

        ax1.imshow(new_im1, aspect="auto")
        ax2.imshow(new_im2, aspect="auto")
        
        ax3.clear()
        ax3.grid(False)
        ax3.set_xlabel("Disparity")
        ax3.set_ylabel("Matching Cost")        
        ax3.plot(np.arange(im1.shape[1]), matching_costs - 20, linewidth=1)
        ax3.margins(x=0, y=0)
        ax3.plot(np.ones((im1.shape[0] + 20))*min_x, np.arange(-20, im1.shape[0]), linewidth=1, c='r')
        ax3.set_xticks([])
        ax3.set_yticks([])
        
        fig.canvas.draw()
        fig.canvas.flush_events()

        plt.show()

    fig, ((ax1, ax2), (ax4, ax3)),  = plt.subplots(2, 2, gridspec_kw={'height_ratios': [1, 0.45]})

    ax1.imshow(im1, aspect="auto")
    ax1.grid(False)
    ax1.axis("off")
    
    ax2.imshow(im2, aspect="auto")
    ax2.grid(False)
    ax2.axis("off")

    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.yaxis.set_ticks_position('left')
    ax3.xaxis.set_ticks_position('bottom')
    ax3.set_xlabel("Disparity")
    ax3.set_ylabel("Matching Cost")
    ax3.grid(False)

    ax4.grid(False)
    ax4.axis("off")

    fig.canvas.mpl_connect('button_press_event', callback)
    fig.tight_layout()

    plt.subplots_adjust(hspace=0.0)    
    plt.show()
    plt.close()

def visualize_pcd(pcd):

    points = pcd.points_list()[0]
    max_x = points[:, 0].max()
    max_y = points[:, 1].max()
    max_z = points[:, 2].max()
    
    R, T = look_at_view_transform(
        dist=float(max_z.item()) + float(max_z.item())/5,
        elev=0, 
        azim=torch.linspace(-45, 45, 12)

    )

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R = R, T = T
    )

    scene = plot_scene({
        "Figure": {
            "PCD": pcd,
            "Camera": cameras,
        }
    })

    scene.show()
    raster_settings = PointsRasterizationSettings(
        image_size=300, 
        radius = 0.003,
        points_per_pixel = 10
    )

    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor(background_color=(1, 1, 1))
    )

    print("Rendering...")
    t0 = time.time()
    images = renderer(pcd.extend(12))
    images = images[:, :, :, :3].numpy()
    final_images = np.zeros((images.shape[0]*2, images.shape[1], images.shape[2], images.shape[3]))
    final_images[:images.shape[0]] = images
    final_images[images.shape[0]:] = np.flip(images, axis=0)
    print("Took {} minutes".format((time.time() - t0)/60.0))

    return final_images

def save_gif(images, output_path, fps=15):
    imageio.mimsave(output_path, images, fps=fps)

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./Art")
    parser.add_argument("--wsize", type=int, default=15)
    args = parser.parse_args()

    data_dir = args.data_dir
    left_im = cv2.imread(os.path.join(data_dir, "view1.png"))
    right_im = cv2.imread(os.path.join(data_dir, "view5.png"))
    left_im = cv2.cvtColor(left_im, cv2.COLOR_BGR2RGB)
    right_im = cv2.cvtColor(right_im, cv2.COLOR_BGR2RGB)
    
    wsize = args.wsize

    match_locs, disparities, left_disp, right_disp = get_disparity(left_im, right_im, wsize)
    pcd = get_depth_from_disparity(match_locs, disparities, left_im)

    images = visualize_pcd(pcd)
    save_gif(images, os.path.join(data_dir, f'wsize{wsize}.gif'), fps=4)
    # visualize_matching(left_im, right_im, wsize)