import math
import os
import sys

import bpy
import numpy as np

from .camera import Camera
from .floor import get_trajectory, plot_floor, show_traj
from .sampler import get_frameidx
from .scene import setup_scene  # noqa
from .tools import delete_objs, load_numpy_vertices_into_blender, mesh_detect
from .vertices import prepare_vertices
from .materials import colored_material_diffuse_BSDF


def create_ego_motion_material():
    """Create a bright material for the ego motion dot."""
    # Bright cyan/blue color for visibility
    return colored_material_diffuse_BSDF(0.0, 0.8, 1.0, a=1, roughness=0.3)


def prepare_ego_motion(ego_motion, downsample=False):
    """
    Prepare ego motion trajectory data for rendering.
    
    Args:
        ego_motion: numpy array of shape [T, 3] containing 3D trajectory
        downsample: whether to downsample the trajectory
    
    Returns:
        Processed ego motion array with axis swap (gravity=Z instead of Y)
    """
    if ego_motion is None:
        return None
    
    ego_data = ego_motion.copy()
    
    # Downsample if needed (same as motion data)
    if downsample:
        ego_data = ego_data[::8]
    
    # Swap axis (gravity=Z instead of Y) - same transform as joints/meshes
    ego_data = ego_data[..., [2, 0, 1]]
    
    # Ensure the trajectory is on or above the floor
    ego_data[..., 2] -= ego_data[..., 2].min()
    
    return ego_data


def create_ego_motion_sphere(location, material, radius=0.08, name="EgoMotionDot"):
    """
    Create a sphere at the given location to represent ego motion position.
    
    Args:
        location: (x, y, z) tuple for sphere position
        material: blender material for the sphere
        radius: radius of the sphere
        name: name for the blender object
    
    Returns:
        Name of the created object
    """
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius,
        location=location,
        segments=16,
        ring_count=8
    )
    obj = bpy.context.object
    obj.name = name
    obj.active_material = material
    bpy.ops.object.shade_smooth()
    return name


def prune_begin_end(data, perc):
    to_remove = int(len(data)*perc)
    if to_remove == 0:
        return data
    return data[to_remove:-to_remove]


def render_current_frame(path):
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(use_viewport=True, write_still=True)



def render(npydata, frames_folder, *, mode, faces_path, gt=False,
           exact_frame=None, num=8, downsample=True,
           canonicalize=True, always_on_floor=False, denoising=True,
           oldrender=True,jointstype="mmm", res="high", init=True,
           accelerator='gpu',device=[0], ego_motion=None):
    if init:
        # Setup the scene (lights / render engine / resolution etc)
        setup_scene(res=res, denoising=denoising, oldrender=oldrender,accelerator=accelerator,device=device)

    is_mesh = mesh_detect(npydata)

    # Put everything in this folder
    if mode == "video":
        if always_on_floor:
            frames_folder += "_of"
        os.makedirs(frames_folder, exist_ok=True)
        # if it is a mesh, it is already downsampled
        if downsample and not is_mesh:
            npydata = npydata[::8]
    elif mode == "sequence":
        img_name, ext = os.path.splitext(frames_folder)
        if always_on_floor:
            img_name += "_of"
        img_path = f"{img_name}{ext}"

    elif mode == "frame":
        img_name, ext = os.path.splitext(frames_folder)
        if always_on_floor:
            img_name += "_of"
        img_path = f"{img_name}_{exact_frame}{ext}"

    # remove X% of begining and end
    # as it is almost always static
    # in this part
    if mode == "sequence":
        perc = 0.2
        npydata = prune_begin_end(npydata, perc)

    # Prepare ego motion data if provided
    ego_motion_prepared = None
    ego_motion_mat = None
    if ego_motion is not None:
        # Apply same downsampling as motion data for video mode
        should_downsample = mode == "video" and downsample and not is_mesh
        ego_motion_prepared = prepare_ego_motion(ego_motion, downsample=should_downsample)
        
        # Also prune for sequence mode
        if mode == "sequence":
            ego_motion_prepared = prune_begin_end(ego_motion_prepared, perc)
        
        ego_motion_mat = create_ego_motion_material()
        print(f"Ego motion prepared with {len(ego_motion_prepared)} frames")

    if is_mesh:
        from .meshes import Meshes
        data = Meshes(npydata, gt=gt, mode=mode,
                      faces_path=faces_path,
                      canonicalize=canonicalize,
                      always_on_floor=always_on_floor)
    else:
        from .joints import Joints
        data = Joints(npydata, gt=gt, mode=mode,
                      canonicalize=canonicalize,
                      always_on_floor=always_on_floor,
                      jointstype=jointstype)

    # Number of frames possible to render
    nframes = len(data)

    # Show the trajectory
    show_traj(data.trajectory)

    # Create a floor - extend it to include ego motion if present
    if ego_motion_prepared is not None:
        # Combine body data and ego motion for floor calculation
        ego_3d = np.expand_dims(ego_motion_prepared, axis=1)  # Shape: [T, 1, 3]
        combined_data = np.concatenate([data.data, ego_3d], axis=1)
        plot_floor(combined_data, big_plane=False)
    else:
        plot_floor(data.data, big_plane=False)

    # initialize the camera with ego motion info for proper framing
    camera = Camera(first_root=data.get_root(0), mode=mode, is_mesh=is_mesh, 
                    ego_motion=ego_motion_prepared)

    frameidx = get_frameidx(mode=mode, nframes=nframes,
                            exact_frame=exact_frame,
                            frames_to_keep=num)

    nframes_to_render = len(frameidx)

    # center the camera to the middle
    if mode == "sequence":
        camera.update(data.get_mean_root())

    imported_obj_names = []
    ego_obj_names = []
    for index, frameidx in enumerate(frameidx):
        if mode == "sequence":
            frac = index / (nframes_to_render-1)
            mat = data.get_sequence_mat(frac)
        else:
            mat = data.mat
            # Get ego position for this frame if available
            ego_pos_for_camera = None
            if ego_motion_prepared is not None and frameidx < len(ego_motion_prepared):
                ego_pos_for_camera = ego_motion_prepared[frameidx]
            camera.update(data.get_root(frameidx), ego_pos=ego_pos_for_camera)

        islast = index == (nframes_to_render-1)

        objname = data.load_in_blender(frameidx, mat)
        
        # Add ego motion sphere if available
        ego_obj_name = None
        if ego_motion_prepared is not None and frameidx < len(ego_motion_prepared):
            ego_pos = tuple(ego_motion_prepared[frameidx])
            ego_obj_name = create_ego_motion_sphere(
                location=ego_pos,
                material=ego_motion_mat,
                radius=0.15,  # Larger radius for visibility
                name=f"EgoMotionDot_{index:04d}"
            )
            if mode == "sequence":
                ego_obj_names.append(ego_obj_name)
        
        name = f"{str(index).zfill(4)}"

        if mode == "video":
            path = os.path.join(frames_folder, f"frame_{name}.png")
        else:
            path = img_path

        if mode == "sequence":
            imported_obj_names.extend(objname)
        elif mode == "frame":
            # Get ego position for frame mode camera update
            ego_pos_for_camera = None
            if ego_motion_prepared is not None and frameidx < len(ego_motion_prepared):
                ego_pos_for_camera = ego_motion_prepared[frameidx]
            camera.update(data.get_root(frameidx), ego_pos=ego_pos_for_camera)

        if mode != "sequence" or islast:
            render_current_frame(path)
            delete_objs(objname)
            if ego_obj_name and mode != "sequence":
                delete_objs([ego_obj_name])

    # bpy.ops.wm.save_as_mainfile(filepath="/Users/mathis/TEMOS_github/male_line_test.blend")
    # exit()

    # remove every object created
    delete_objs(imported_obj_names)
    delete_objs(ego_obj_names)
    delete_objs(["Plane", "myCurve", "Cylinder", "EgoMotionDot"])

    if mode == "video":
        return frames_folder
    else:
        return img_path
