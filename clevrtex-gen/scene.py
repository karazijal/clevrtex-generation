import bpy

CAMERA = 'Camera'
KEYLAMP = 'Lamp_Key'
BACKLAMP = 'Lamp_Back'
FILLLIGHT = 'Lamp_Fill'
GROUND = 'Ground'

class SceneWrapper:
    def __init__(self, scene_base_file):
        bpy.ops.wm.open_mainfile(filepath=scene_base_file)

        self._objects = bpy.data.objects
        assert CAMERA in self._objects
        assert KEYLAMP in self._objects
        assert BACKLAMP in self._objects
        assert FILLLIGHT in self._objects
        assert GROUND in self._objects

        self._ref = bpy.context.scene
        self._ref.use_nodes = True  # Turn on output compositor

        self.node_tree = self._ref.node_tree
        self.view_layer = self._ref.view_layers[0]

        self.output_layers_node = None
        for n in self.node_tree.nodes:
            if n.name == 'Render Layers':
                self.output_layers_node = n
                break
        else:
            raise RuntimeError("Could not locate root Compositor node for layer output")

    @property
    def ground(self):
        return self._objects[GROUND]

    @property
    def camera(self):
        return self._objects[CAMERA]

    @property
    def keylamp(self):
        return self._objects[KEYLAMP]

    @property
    def backlamp(self):
        return self._objects[BACKLAMP]

    @property
    def filllight(self):
        return self._objects[FILLLIGHT]
