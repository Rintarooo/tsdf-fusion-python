"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import time

import cv2
import numpy as np

import fusion

def modify_depthmap(src):
  depth_min, depth_max = 0.5, 6.0
  near, far = 1./depth_min, 1./depth_max
  minv, maxv = far, near
  h, w = src.shape
  dst = np.zeros_like(src)

  # https://github.com/Rintarooo/dtam/blob/1272fad70d92228d4706a46545980637d748082c/Viewer3D/pcl3d.cpp#L185-L186
  for v in range(h):
    for u in range(w):
      srcv = src[v,u]
      inv_depth_val = (srcv * (maxv - minv))/65535. + minv
      if(inv_depth_val != 0): dst[v,u] = 1./inv_depth_val       
  return dst

if __name__ == "__main__":
  # ======================================================================================================== #
  # (Optional) This is an example of how to compute the 3D bounds
  # in world coordinates of the convex hull of all camera view
  # frustums in the dataset
  # ======================================================================================================== #
  print("Estimating voxel volume bounds...")
  n_imgs = 50
  cam_intr = np.loadtxt("input/icl_nuim/of_kt0/data/camera-intrinsics.txt", delimiter=' ')
  vol_bnds = np.zeros((3,2))
  
  print("cam_intr: ", cam_intr)
  # for i in range(0, n_imgs, 1):
  for i in range(1, n_imgs, 1):
    print("i: ", i)
    # Read depth image and camera pose
    # depth_im = cv2.imread("data/frame-%06d.depth.png"%(i),-1).astype(float)
    
    # depth_im = cv2.imread("input/icl_nuim/of_kt0/depth_v0/%d.png"%(i),-1).astype(float)
    # depth_im /= 5000.  # depth is saved in 16-bit PNG in millimeters
    depth_im = cv2.imread("input/icl_nuim/of_kt0/depth_v1/%d.png"%(i),-1).astype(float)

    depth_im = modify_depthmap(depth_im)
    ## debug
    # depth_max = np.max(depth_im)
    # depth_min = np.min(depth_im)
    # print("depth_min: {}\ndepth_max: {}".format(depth_min, depth_max))

    # depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
    # depth_im[depth_im == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
    # cam_pose = np.loadtxt("data/frame-%06d.pose.txt"%(i))  # 4x4 rigid transformation matrix
    cam_pose = np.loadtxt("input/icl_nuim/of_kt0/data/pose/%d.txt"%(i))  # 4x4 rigid transformation matrix


    # Compute camera view frustum and extend convex hull
    view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
    vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
    vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))
    # print("vol_bnds: ", vol_bnds)
  # ======================================================================================================== #

  # ======================================================================================================== #
  # Integrate
  # ======================================================================================================== #
  # Initialize voxel volume
  print("Initializing voxel volume...")
  tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.02)
  # tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.5)

  # Loop through RGB-D images and fuse them together
  t0_elapse = time.time()
  # for i in range(0, n_imgs, 1):
  for i in range(1, n_imgs, 1):
    print("Fusing frame %d/%d"%(i+1, n_imgs))

    # Read RGB-D image and camera pose
    color_image = cv2.cvtColor(cv2.imread("input/icl_nuim/of_kt0/rgb/%d.png"%(i)), cv2.COLOR_BGR2RGB)
    
    # depth_im = cv2.imread("data/frame-%06d.depth.png"%(i),-1).astype(float)
    depth_im = cv2.imread("input/icl_nuim/of_kt0/depth_v1/%d.png"%(i),-1).astype(float)
    # depth_im /= 1000.
    # depth_im[depth_im == 65.535] = 0
    depth_im = modify_depthmap(depth_im)


    # cam_pose = np.loadtxt("data/frame-%06d.pose.txt"%(i))
    cam_pose = np.loadtxt("input/icl_nuim/of_kt0/data/pose/%d.txt"%(i))

    # Integrate observation into voxel volume (assume color aligned with depth)
    tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)

  fps = n_imgs / (time.time() - t0_elapse)
  print("Average FPS: {:.2f}".format(fps))

  # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving mesh to mesh.ply...")
  verts, faces, norms, colors = tsdf_vol.get_mesh()
  fusion.meshwrite("mesh.ply", verts, faces, norms, colors)

  # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving point cloud to pc.ply...")
  point_cloud = tsdf_vol.get_point_cloud()
  fusion.pcwrite("pc.ply", point_cloud)