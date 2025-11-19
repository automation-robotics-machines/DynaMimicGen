import numpy as np

from robosuite.models.objects import MujocoXMLObject

from dmg.models import assets_root

class TargetObject(MujocoXMLObject):
    """
    target object 
    """

    def __init__(self, name):
        
        model_path = assets_root
        model_path += "/objects/target_pose.xml"
        super().__init__(
            model_path,
            name=name,
            joints=None,
            # joints=[dict(type="hinge", limited="true", axis="0 0 1", range="-3 3")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

class CSL(MujocoXMLObject):
    """
    cable suspended load 
    """
    
    def __init__(self, name):
        
        model_path = assets_root
        model_path += "/objects/cable_suspended_load.xml"
        
        super().__init__(
            model_path,
            name=name,
            joints=None,
            obj_type="all",
            duplicate_collision_geoms=True,
        )
        