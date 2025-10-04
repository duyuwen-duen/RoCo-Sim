"""
Modify the car paint color.
"""
import bpy
import os
import json
import torch


def modify_car_color(model:bpy.types.Object, material_key, color):
    """
    Args:
        model: bpy_types.Objct
            car model
        material_key: str
            key name in model.material_slots. Refer to the car paint material.
        color: list of float
            target base color, [R,G,B,alpha] 
    """
    material = model.material_slots[material_key].material
    # Modifiy Metaillic, Specular, Roughness if needed
    # Suppose use Principled BSDF
    material.node_tree.nodes["Principled BSDF"].inputs['Base Color'].default_value = color


def set_model_params(loc, rot, rot_mode="XYZ", model_obj_name="Car", target_color=None):
    """
    Args:
        loc: list
            [x, y, z]
        rot: list
            [angle1, angle2, angle3] (rad.)
        rot_mode: str
            Euler angle order
        model_obj_name: str
            name of the entire model. New obj name.
        target_color: dict (optinoal)
            {"material_key":.., "color": ...}
    """
    model = bpy.data.objects[model_obj_name]
    model.location = loc
    model.rotation_mode = rot_mode
    model.rotation_euler = rot
    if target_color is not None and target_color['color'] is not [0,0,0,1]:
        modify_car_color(model, 
                         target_color['material_key'], 
                         target_color['color']
        )


def rotate_axis_angle(points, axis, angle_degrees):
    """
    旋转一组点。
    
    参数:
    - points: (N, 3) 形状的张量，表示点集。
    - axis: 旋转轴，(3,) 形状的张量或列表。
    - angle_degrees: 旋转角度（度）。
    
    返回:
    - 旋转后的点集。
    """
    # 将角度转换为弧度
    angle_radians = torch.tensor(angle_degrees) * (torch.pi / 180.0)
    
    # 确保旋转轴是单位向量
    axis = torch.tensor(axis)
    axis = axis / torch.norm(axis)
    
    # 计算罗德里格斯旋转矩阵的组成部分
    cos_theta = torch.cos(angle_radians)
    sin_theta = torch.sin(angle_radians)
    I = torch.eye(3)
    outer = torch.outer(axis, axis)
    A = torch.tensor([[0, -axis[2], axis[1]], 
                      [axis[2], 0, -axis[0]], 
                      [-axis[1], axis[0], 0]])
    
    # 组装旋转矩阵
    R = cos_theta * I + (1 - cos_theta) * outer + sin_theta * A
    
    # 应用旋转
    rotated_points = torch.matmul(points, R)
    
    return rotated_points

def add_model_params_new(model_setting):
    """
    model_setting includes:
        - blender_file: path to object model file
        - insert_pos: list of len 3
        - insert_rot: list of len 3 
        - model_obj_name: object name within blender_file
        - new_obj_name: object name in this scene
        - target_color: optional .
    """
    model_obj_name = model_setting['model_obj_name']
    new_obj_name = model_setting['new_obj_name']
    target_color = model_setting.get('target_color', None)

    if model_setting['type'] == 'car':
        blender_file = model_setting['blender_file']
        with bpy.data.libraries.load(blender_file, link=False) as (data_from, data_to):
            data_to.objects = data_from.objects

        # link to context is required! 这是将实际对象添加到场景的关键步骤。
        for obj in data_to.objects: 
            if obj.name == model_obj_name:
                bpy.context.collection.objects.link(obj)
    
    elif model_setting['type'] == 'human':
        model_path = model_setting['smpl_file']
        from smplx import SMPL 
        import numpy as np
        import torch
        smpl = SMPL(model_path, gender='male', batch_size=1)
        betas = np.zeros((1, 10))  # SMPL模型的形状参数
        pose = model_setting['pose']
        print(f"==================== pose: {pose.shape} =====================")
        betas_tensor = torch.tensor(betas, dtype=torch.float32)
        pose_tensor = torch.tensor(pose, dtype=torch.float32).reshape(1, -1)
        
        output = smpl(betas=betas_tensor, body_pose=pose_tensor[:,3:], global_orient=pose_tensor[:,:3])

        vertices = output.vertices.detach()
        faces = torch.from_numpy(np.int32(smpl.faces))
        faces_uvs = torch.load('/home/ubuntu/dingjuwang/traj_play/render/utils/blender_utils/model/uv/faces_uvs.pt')
        verts_uvs = torch.load('/home/ubuntu/dingjuwang/traj_play/render/utils/blender_utils/model/uv/verts_uvs.pt')

        vertices = rotate_axis_angle(vertices, [1.,0.,0.], 270)
        vertices = rotate_axis_angle(vertices, [0.,1.,0.], 0)
        vertices = rotate_axis_angle(vertices, [0.,0.,1.], 90)

        ## for pacer+ coordinate
        
        vertices = torch.cat((vertices[:,:,2:3], vertices[:,:,1:2], -vertices[:,:,0:1]), dim=-1)
        
        vertices += model_setting['trans'] * [1,1,1]
        vertices[..., -1] = vertices[..., -1] - torch.min(vertices[...,-1])
        mesh_data = bpy.data.meshes.new("new_mesh")
        mesh_data.from_pydata(vertices[0], [], faces)
        mesh_data.update()

        mesh_object = bpy.data.objects.new("human_mesh", mesh_data)
        bpy.context.collection.objects.link(mesh_object)

        mesh_data.uv_layers.new(name="new_uv_map")
        uv_layer = mesh_data.uv_layers.active.data
        verts_uvs_list = verts_uvs.tolist()
        
        # 为每个面的每个顶点设置UV坐标
        for poly in mesh_data.polygons:
            for loop_index, vert_index in zip(poly.loop_indices, faces_uvs[poly.index]):
                uv = (verts_uvs_list[vert_index])
                uv_layer[loop_index].uv = uv

        human_texture_image_path = model_setting['human_texture_file']
        human_texture_image = bpy.data.images.load(human_texture_image_path)

        cloth_texture_image_path = model_setting['cloth_texture_file']
        cloth_texture_image = bpy.data.images.load(cloth_texture_image_path)


        mat = bpy.data.materials.new(name="HumanClothMaterial")
        mesh_object.data.materials.append(mat)

        # 使用节点
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        # 创建节点
        texture_human = nodes.new('ShaderNodeTexImage')
        texture_human.image = human_texture_image
        texture_cloth = nodes.new('ShaderNodeTexImage')
        texture_cloth.image = cloth_texture_image

        mix_node = nodes.new('ShaderNodeMixRGB')
        mix_node.blend_type = 'MIX'
        mix_node.inputs['Fac'].default_value = 1.0  # 使用布料图的Alpha作为因子

        # 连接节点
        links.new(texture_cloth.outputs['Alpha'], mix_node.inputs['Fac'])
        links.new(texture_human.outputs['Color'], mix_node.inputs[1])
        links.new(texture_cloth.outputs['Color'], mix_node.inputs[2])

        shader = nodes.new('ShaderNodeBsdfPrincipled')
        output_node = nodes.new('ShaderNodeOutputMaterial')

        links.new(mix_node.outputs['Color'], shader.inputs['Base Color'])
        links.new(shader.outputs['BSDF'], output_node.inputs['Surface'])
        
    if model_obj_name in bpy.data.objects:
        imported_object = bpy.data.objects[model_obj_name]
        imported_object.name = new_obj_name

    # rename material
    for slot in imported_object.material_slots:
        material = slot.material
        if material:
            # 为每个材质添加前缀
            material.name = new_obj_name + "_" + material.name

    if target_color is not None:
        target_color['material_key'] = new_obj_name + "_" + target_color['material_key']


    set_model_params(model_setting['insert_pos'],
                     model_setting['insert_rot'], 
                     rot_mode="XYZ", 
                     model_obj_name=new_obj_name, 
                     target_color=target_color)


def add_model_params(model_setting):
    """
    model_setting includes:
        - blender_file: path to object model file
        - insert_pos: list of len 3
        - insert_rot: list of len 3 
        - model_obj_name: object name within blender_file
        - new_obj_name: object name in this scene
        - target_color: optional .
    """
    blender_file = model_setting['blender_file']
    model_obj_name = model_setting['model_obj_name']
    new_obj_name = model_setting['new_obj_name']
    target_color = model_setting.get('target_color', None)

    # append object into the scene, use bpy.ops.wm.append. 
    # not work in non batch mode (wo -b)，会遇到上下文的bug
    # inner_path = 'Object'
    # bpy.ops.wm.append(
    #     filepath=os.path.join(blender_file, inner_path, model_obj_name),
    #     directory=os.path.join(blender_file, inner_path),
    #     filename=model_obj_name,
    # )

    # append object into the scene, use bpy.data.libraries.load
    # 这一步仅仅是将从外部.blend文件中加载的对象数据复制到了data_to 上下文，但尚未将这些对象添加到任何场景。
    # 相关材质和贴图也被自动load了，不用手动重复load
    with bpy.data.libraries.load(blender_file, link=False) as (data_from, data_to):
        data_to.objects = data_from.objects

    # link to context is required! 这是将实际对象添加到场景的关键步骤。
    for obj in data_to.objects: 
        if obj.name == model_obj_name:
            bpy.context.collection.objects.link(obj)
            
    if model_obj_name in bpy.data.objects:
        imported_object = bpy.data.objects[model_obj_name]
        imported_object.name = new_obj_name

    # rename material
    for slot in imported_object.material_slots:
        material = slot.material
        if material:
            # 为每个材质添加前缀
            material.name = new_obj_name + "_" + material.name

    if target_color is not None:
        target_color['material_key'] = new_obj_name + "_" + target_color['material_key']


    set_model_params(model_setting['insert_pos'],
                     model_setting['insert_rot'], 
                     rot_mode="XYZ", 
                     model_obj_name=new_obj_name, 
                     target_color=target_color)


def add_plane(car_obj):
    # we need create a plane for the model
    # size = max(car_obj['size'][0], car_obj['size'][1])
    bpy.ops.mesh.primitive_plane_add(size=1)

    if hasattr(bpy.context, 'object'): # background mode
        plane = bpy.context.object
    else:   # interface mode
        plane = bpy.data.objects["Plane"]
 
    plane.location = [car_obj['insert_pos'][0], car_obj['insert_pos'][1], car_obj['insert_pos'][2]]
    plane.rotation_euler = car_obj['insert_rot']
    plane.scale = (car_obj['size'][1]*1.2, car_obj['size'][0]*1.2,1)
    plane.name =  "plane"
    plane.is_shadow_catcher = True

    # new material for the plane
    material = bpy.data.materials.new(name="new_plane_material")
    plane.data.materials.append(material)

    # set material color for the plane
    material.use_nodes = True
    nodes = material.node_tree.nodes
    BSDF_node = nodes.get("Principled BSDF")

    if BSDF_node:
        # base color default dark. will not affect shadow color, but will affect the reflection on the car
        BSDF_node.inputs[0].default_value = (0.004, 0.005, 0.006, 1) 
        BSDF_node.inputs[9].default_value = 1  # roughness
        BSDF_node.inputs[21].default_value = 1  # alpha. Otherwise composition with background will lead to strong black.