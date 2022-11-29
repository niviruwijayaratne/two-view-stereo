import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time
import open3d as o3d

b = 160 #baseline, mm
f = 3740 #focal length, pixels
data_dir = "./Laundry"

#Note images already rectified
def get_disparity(left_im, right_im, wsize, min_disp=10):
    t0 = time.time()
    H = left_im.shape[0]
    W = left_im.shape[1]

    padded_left_im = np.pad(left_im, ((wsize//2, wsize//2), (wsize//2, wsize//2), (0, 0)))
    padded_right_im = np.pad(right_im, ((wsize//2, wsize//2), (wsize//2, wsize//2), (0, 0)))
    match_locs = np.zeros((H*W, 2, 2))
    
    pixel_counter = 0
    for y in range(H):    
        for x_l in range(W):
            print("Pixel at {}, {}".format(y, x_l))
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

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.colors = o3d.utility.Vector3dVector(colors/255.0)

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

if __name__ == '__main__':
    left_im = cv2.imread(os.path.join(data_dir, "view1.png"))
    right_im = cv2.imread(os.path.join(data_dir, "view5.png"))
    left_im = cv2.cvtColor(left_im, cv2.COLOR_BGR2RGB)
    right_im = cv2.cvtColor(right_im, cv2.COLOR_BGR2RGB)
    wsize = 21

    match_locs, disparities, left_disp, right_disp = get_disparity(left_im, right_im, wsize)
    pcd = get_depth_from_disparity(match_locs, disparities, left_im)
    o3d.visualization.draw_geometries([pcd])
    # visualize_matching(left_im, right_im, wsize)