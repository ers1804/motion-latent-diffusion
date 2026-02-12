import bpy
import numpy as np


class Camera:
    def __init__(self, *, first_root, mode, is_mesh, ego_motion=None):
        camera = bpy.data.objects['Camera']

        ## initial position
        camera.location.x = 7.36
        camera.location.y = -6.93
        if is_mesh:
            # camera.location.z = 5.45
            camera.location.z = 5.6
        else:
            camera.location.z = 5.2

        # wider point of view
        if mode == "sequence":
            if is_mesh:
                camera.data.lens = 65
            else:
                camera.data.lens = 85
        elif mode == "frame":
            if is_mesh:
                camera.data.lens = 130
            else:
                camera.data.lens = 85
        elif mode == "video":
            if is_mesh:
                camera.data.lens = 110
            else:
                # avoid cutting person
                camera.data.lens = 85
                # camera.data.lens = 140

        # If ego motion is provided, zoom out more and position to see both
        if ego_motion is not None:
            # Reduce lens (wider FOV) to capture both body and ego motion
            camera.data.lens = min(camera.data.lens, 50)
            # Move camera back and up for better overview
            camera.location.z += 4.0
            camera.location.x += 3.0
            camera.location.y -= 3.0

        # camera.location.x += 0.75

        self.mode = mode
        self.camera = camera
        self.ego_motion = ego_motion

        self.camera.location.x += first_root[0]
        self.camera.location.y += first_root[1]

        self._root = first_root

    def update(self, newroot, ego_pos=None):
        """Update camera position to follow the action.
        
        Args:
            newroot: New root position of the body/skeleton
            ego_pos: Optional ego motion position to also track
        """
        # Always track the SMPL body root, even when ego motion is present
        delta_root = newroot[:2] - self._root[:2]
        self._root = newroot

        self.camera.location.x += delta_root[0]
        self.camera.location.y += delta_root[1]
