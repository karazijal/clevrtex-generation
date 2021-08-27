from pathlib import Path

import bpy
import bpy_extras


def get_camera_coords(cam, pos):
    """
    For a specified point, get both the 3D coordinates and 2D pixel-space
    coordinates of the point from the perspective of the camera.

    Inputs:
    - cam: Camera object
    - pos: Vector giving 3D world-space position

    Returns a tuple of:
    - (px, py, pz): px and py give 2D image-space coordinates; pz gives depth
      in the range [-1, 1]
    """
    x, y, z = bpy_extras.object_utils.world_to_camera_view(bpy.context.scene, cam, pos)
    scale = bpy.context.scene.render.resolution_percentage / 100.0
    w = int(scale * bpy.context.scene.render.resolution_x)
    h = int(scale * bpy.context.scene.render.resolution_y)
    px = int(round(x * w))
    py = int(round(h - y * h))
    return (px, py, z)


def configure_cycles(output_path, width, height, tile_size, num_samples, min_bounces, max_bounces, use_gpu=False):
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.render.filepath = str(output_path)
    bpy.context.scene.render.resolution_x = width
    bpy.context.scene.render.resolution_y = height
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.tile_x = tile_size
    bpy.context.scene.render.tile_y = tile_size
    if use_gpu:
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        bpy.context.scene.cycles.device = 'GPU'
        for d in bpy.context.preferences.addons["cycles"].preferences.devices:
            d["use"] = 1

    bpy.data.worlds['World'].cycles.sample_as_light = True
    bpy.context.scene.cycles.blur_glossy = 2.0
    bpy.context.scene.cycles.samples = num_samples
    bpy.context.scene.cycles.transparent_min_bounces = min_bounces
    bpy.context.scene.cycles.transparent_max_bounces = max_bounces


def compositor_output(out_socket, node_tree, output_path_prefix, set_bw=False, mkdir=True, frame_id=1):
    out_dir = Path(output_path_prefix).parent
    if mkdir:
        out_dir.mkdir(parents=True, exist_ok=True)
    elif not out_dir.exists() or not out_dir.is_dir():
        raise RuntimeError(f'Cannot output to {out_dir}')

    out_name = Path(output_path_prefix).name

    output_node = node_tree.nodes.new('CompositorNodeOutputFile')
    output_node.base_path = str(out_dir)
    output_node.file_slots[0].path = str(out_name)

    if set_bw:
        output_node.format.color_mode = 'BW'
        output_node.file_slots[0].format.color_mode = 'BW'

    node_tree.links.new(
        out_socket,
        output_node.inputs['Image']
    )
    return out_dir / f"{out_name}{frame_id:0>4d}.png"


def compositor_obj_mask_output(out_socket, node_tree, obj_index, output_path_prefix, mkdir=True, frame_id=1):
    id_mask_node = node_tree.nodes.new('CompositorNodeIDMask')
    id_mask_node.index = obj_index
    id_mask_node.use_antialiasing = False
    node_tree.links.new(
        out_socket,
        id_mask_node.inputs['ID value']
    )
    output_path_prefix = str(output_path_prefix)
    if output_path_prefix.endswith('.png'):
        output_path_prefix = output_path_prefix[:-4]
    output_path_prefix += f'_o{obj_index:0>2d}_'
    return compositor_output(id_mask_node.outputs['Alpha'], node_tree, output_path_prefix, set_bw=True, mkdir=mkdir,
                             frame_id=frame_id)


def add_object_geometry(object_path, scale, loc, rotation=0):
    """
    Load an object from a file. object_path points to .blend file with
    object_path.stem object inside the scene, with scale 0 and positions at the origin

    - scale: scalar giving the size that the object should be in the scene
    - loc: tuple (x, y) location on the ground plane
    - rotation: scalar rotation to apply to the object
    """
    # First figure out how many of this object are already in the scene so we can
    # give the new object a unique name
    object_path = Path(object_path)
    name = object_path.stem
    count = 0
    for obj in bpy.data.objects:
        if obj.name.startswith(name):
            count += 1

    filename = object_path / 'Object' / name
    bpy.ops.wm.append(filename=str(filename))

    # Give it a new name to avoid conflicts
    new_name = f'{name}_{count}'
    bpy.data.objects[name].name = new_name

    # Set the new object as active, then rotate, scale, and translate it
    x, y = loc
    o = bpy.data.objects[new_name]
    o.select_set(state=True, view_layer=bpy.context.view_layer)
    bpy.context.view_layer.objects.active = o
    bpy.context.object.rotation_euler[2] = rotation  # Rotate around z
    bpy.ops.transform.resize(value=(scale, scale, scale))
    bpy.ops.transform.translate(value=(x, y, scale))
    o.select_set(state=False, view_layer=bpy.context.view_layer)


def load_material(path):
    path = Path(path)
    filepath = path / 'NodeTree' / path.stem
    bpy.ops.wm.append(filename=str(filepath))


def add_material(obj, mat_path, **properties):
    """
    Create a new material and assign it to the active object. "name" should be the
    name of a material that has been previously loaded using load_materials.
    """

    # Sometime Displacement is called Displacement Strength
    if 'Displacement' in properties:
        properties['Displacement Strength'] = properties['Displacement']

    # Figure out how many materials are already in the scene
    mat_count = len(bpy.data.materials)
    names = {m.name for m in bpy.data.materials}
    name = mat_path.stem
    mat_name = mat_path.stem
    if name in names:
        idx = sum(1 for m in bpy.data.materials if m.name.startswith(name))
        mat_name = name + f'_{idx + 1}'

    # Create a new material
    mat = bpy.data.materials.new(mat_name)
    mat.name = mat_name
    mat.use_nodes = True
    mat.cycles.displacement_method = 'BOTH'

    # Attach the new material to the object
    # Make sure it doesn't already have materials
    assert len(
        obj.data.materials) == 0, f"{obj.name} has multiple materials ({', '.join(m.name for m in obj.data.materials if m is not None)}), adding {name} will fail"
    obj.data.materials.append(mat)

    mat.node_tree.links.clear()
    mat.node_tree.nodes.clear()

    output_node = mat.node_tree.nodes.new('ShaderNodeOutputMaterial')
    output_node.is_active_output = True

    # Add a new GroupNode to the node tree of the active material,

    group_node = mat.node_tree.nodes.new('ShaderNodeGroup')
    if name not in bpy.data.node_groups:
        load_material(mat_path)
    group_node.node_tree = bpy.data.node_groups[name]
    # Also this seems to be the only way to copy a node tree in the headless mode

    # Wire first by-name then by preset names, to the group outputs to the material output
    for out_socket in group_node.outputs:
        if out_socket.name in output_node.inputs:
            mat.node_tree.links.new(
                group_node.outputs[out_socket.name],
                output_node.inputs[out_socket.name],
            )
        else:
            # print(f"{out_socket.name} not found in the output of the material")
            pass
        if not output_node.inputs['Surface'].is_linked:
            if 'Shader' in group_node.outputs and not group_node.outputs['Shader'].is_linked:
                # print(f"Unlinked Surface socket in the material output; trying to fill with Shader socket of the group")
                mat.node_tree.links.new(
                    group_node.outputs["Shader"],
                    output_node.inputs["Surface"],
                )
            elif 'BSDF' in group_node.outputs and not group_node.outputs['BSDF'].is_linked:
                # print(f"Unlinked Surface socket in the material output; trying to fill with BSDF socket of the group")
                mat.node_tree.links.new(
                    group_node.outputs["BSDF"],
                    output_node.inputs["Surface"],
                )
            else:
                raise ValueError(f"Cannot resolve material output for {mat.name}")

    for inp in group_node.inputs:
        if inp.name in properties:
            inp.default_value = properties[inp.name]

    return mat


def add_shadeless_nodes_to_material(mat, shadeless_clr):
    """
    Inject nodes to the material tree required for rendering solid colour output
    """
    mix_node = mat.node_tree.nodes.new('ShaderNodeMixShader')
    mix_node.name = 'InjectedShadelessMix'
    dif_node = mat.node_tree.nodes.new('ShaderNodeBsdfDiffuse')
    dif_node.name = 'InjectedShadelessDif'
    lit_node = mat.node_tree.nodes.new('ShaderNodeLightPath')
    lit_node.name = 'InjectedShadelessLit'
    emi_node = mat.node_tree.nodes.new('ShaderNodeEmission')
    emi_node.name = 'InjectedShadelessEmission'
    emi_node.inputs['Color'].default_value = shadeless_clr
    l1 = mat.node_tree.links.new(
        lit_node.outputs['Is Camera Ray'],
        mix_node.inputs['Fac'],
    )
    l2 = mat.node_tree.links.new(
        dif_node.outputs['BSDF'],
        mix_node.inputs[1],
    )
    l3 = mat.node_tree.links.new(
        emi_node.outputs['Emission'],
        mix_node.inputs[2],
    )
    return mix_node, (mix_node, dif_node, lit_node, emi_node), (l1, l2, l3)


def set_to_shadeless(mat, shadeless_clr):
    """
    Rewire the material to output solid colour <shadeless_clr>.
    Returns a callback that returns the material to original state
    """
    # print(f"Setting {mat.name} to shadeless")
    # Locate output node
    output_node = None
    for n in mat.node_tree.nodes:
        if n.name == 'Material Output':
            output_node = n
            break
    else:
        raise ValueError(f"Could not locate output node in {mat.name} Material")

    socket = None
    if output_node.inputs['Surface'].is_linked:
        l = None
        for link in mat.node_tree.links:
            if link.to_socket == output_node.inputs['Surface']:
                l = link
                break
        else:
            raise ValueError(f"Could not locate output node Surface link in {mat.name} Material")
        socket = l.from_socket.node.outputs[l.from_socket.name]  # Lets hope there not multiple with the same name
        # print(f"Will try to restore link between {l.from_socket.node.name}:{l.from_socket.name} {socket}")
        mat.node_tree.links.remove(l)

    # Check that shadeless rendering nodes have not already been injected to this material
    mix_node = None
    for n in mat.node_tree.nodes:
        if n.name == 'InjectedShadelessMix':
            mix_node = n
            break
    if mix_node is None:
        # Inject the nodes
        mix_node, nodes, links = add_shadeless_nodes_to_material(mat, shadeless_clr)
    else:
        # If they already exist; just set the colour to the correct value
        nodes = []
        for n in mat.node_tree.nodes:
            if n.name == 'InjectedShadelessEmission':
                n.inputs['Color'].default_value = shadeless_clr
            if n.name.startswith('InjectedShadeless'):
                nodes.append(n)
        links = set()
        for n in nodes:
            for s in n.inputs:
                if s.is_linked:
                    for l in s.links:
                        links.add(l)

    # Check and correct the node connection
    if mix_node.outputs['Shader'].is_linked:
        # Check and reset the node links between Shadeless mix node and the material output
        offending_link = None
        for link in mat.node_tree.links:
            if link.from_socker.node == mix_node:
                offending_link = link
                break
        else:
            raise ValueError(f"Could not locate offending mix_shader link in the {mat.name} Material")
        mat.node_tree.links.remove(offending_link)

    temp_link = mat.node_tree.links.new(
        mix_node.outputs['Shader'],
        output_node.inputs['Surface'],
    )

    def undo_callback():
        mat.node_tree.links.remove(temp_link)
        if socket:
            mat.node_tree.links.new(socket, output_node.inputs['Surface'])
        # print(f"Reverting {mat.name} to the original")
        for l in links:
            mat.node_tree.links.remove(l)
        for n in nodes:
            mat.node_tree.nodes.remove(n)

    return undo_callback


def dump_mat(mat):
    print(f"Material {mat.name}")
    nt = mat.node_tree
    for n in nt.nodes:
        print('\t', n.name, n.label, )
        for i in n.inputs:
            print('\t\t>', i.name,
                  f'<{i.links[0].from_socket.node.name}:{i.links[0].from_socket.name}' if len(i.links) else '')
