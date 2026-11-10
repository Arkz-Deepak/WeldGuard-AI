"""
AutoTwin-AI v2.0: Spatiotemporal Video Dataset Generator
Generates 30 FPS frame sequences in Blender simulating robotic GMAW welding
with camera tracking Tool Center Point (TCP), animated weld pool, arc flicker,
procedural spatter particles, and injection of spatiotemporal anomalies.
"""

import json
import math
import os
import random
import sys
import time
from pathlib import Path

import bpy

# ---------------------------------------------------------
# Configuration & Environment
# ---------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
OBJ_PATH = ROOT_DIR / "cad_model" / "28000.obj"
DATASET_DIR = ROOT_DIR / "synthetic_dataset" / "video_sequences"

# Parse CLI arguments if passed after '--' (e.g. blender -b -P script.py -- --normal 10)
cli_args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []

def get_arg(name, default_val):
    if f"--{name}" in cli_args:
        idx = cli_args.index(f"--{name}")
        if idx + 1 < len(cli_args):
            return cli_args[idx + 1]
    return os.environ.get(name.upper(), str(default_val))

NUM_NORMAL_SEQS = int(get_arg("num_normal_seqs", 5))
NUM_SPATTER_SEQS = int(get_arg("num_spatter_seqs", 2))
NUM_ARCDROP_SEQS = int(get_arg("num_arcdrop_seqs", 2))
NUM_JERK_SEQS = int(get_arg("num_jerk_seqs", 2))
FRAMES_PER_SEQ = int(get_arg("frames_per_seq", 30))
RENDER_RESOLUTION = int(get_arg("render_resolution", 256))
RENDER_SAMPLES = int(get_arg("render_samples", 12))
FPS = int(get_arg("fps", 30))

RANDOM_SEED = int(get_arg("random_seed", 2026))
random.seed(RANDOM_SEED)


# Weld seam trajectory across Front-Left Suspension Joint
SEAM_START = (-1.55, -0.303, -0.075)
SEAM_END = (-1.41, -0.303, -0.075)


def clean_scene():
    """Remove all objects and unlinked data blocks."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras):
        for block in collection:
            if block.users == 0:
                collection.remove(block)


def setup_render_engine():
    """Configure Cycles GPU OptiX / CUDA rendering with denoising."""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = RENDER_RESOLUTION
    scene.render.resolution_y = RENDER_RESOLUTION
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.fps = FPS
    scene.cycles.samples = RENDER_SAMPLES
    scene.cycles.max_bounces = 2
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transparent_max_bounces = 1
    scene.cycles.transmission_bounces = 0
    scene.cycles.volume_bounces = 0
    scene.render.use_persistent_data = True

    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        for dev_type in ('OPTIX', 'CUDA'):
            try:
                prefs.compute_device_type = dev_type
                prefs.get_devices()
                gpu_found = False
                for d in prefs.devices:
                    if d.type != 'CPU':
                        d.use = True
                        gpu_found = True
                    else:
                        d.use = False
                if gpu_found:
                    scene.cycles.device = 'GPU'
                    scene.cycles.use_denoising = True
                    scene.cycles.denoiser = 'OPTIX' if dev_type == 'OPTIX' else 'OPENIMAGEDENOISE'
                    print(f"[Blender] {dev_type} GPU acceleration enabled")
                    break
            except Exception:
                continue
    except Exception as e:
        print("[Blender] Device init notice:", e)


def import_and_setup_chassis():
    """Import 28000.obj and apply PBR steel material."""
    bpy.ops.wm.obj_import(filepath=str(OBJ_PATH))
    imported = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
    if not imported:
        raise RuntimeError(f"No mesh objects found in {OBJ_PATH}")
    if len(imported) > 1:
        bpy.context.view_layer.objects.active = imported[0]
        bpy.ops.object.join()
    chassis = bpy.context.active_object
    chassis.name = 'CadChassis'
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    chassis.location = (0.0, 0.0, 0.0)
    chassis.rotation_euler = (0.0, 0.0, 0.0)

    max_dim = max(chassis.dimensions)
    if max_dim > 0:
        scale_val = 4.0 / max_dim
        chassis.scale = (scale_val, scale_val, scale_val)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

    # Steel material
    mat = bpy.data.materials.new(name='ChassisSteel')
    bsdf = mat.node_tree.nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = (0.28, 0.29, 0.31, 1.0)
    bsdf.inputs['Metallic'].default_value = 0.75
    bsdf.inputs['Roughness'].default_value = 0.28
    chassis.data.materials.clear()
    chassis.data.materials.append(mat)
    return chassis


def create_world_and_lighting():
    """Set ambient dark industrial factory background and sun."""
    scene = bpy.context.scene
    world = scene.world.node_tree
    world.nodes.clear()
    bg = world.nodes.new(type='ShaderNodeBackground')
    bg.inputs['Color'].default_value = (0.04, 0.04, 0.05, 1.0)
    bg.inputs['Strength'].default_value = 0.8
    out = world.nodes.new(type='ShaderNodeOutputWorld')
    world.links.new(bg.outputs['Background'], out.inputs['Surface'])

    # Low ambient sun
    sun_data = bpy.data.lights.new(name="AmbientSun", type='SUN')
    sun_obj = bpy.data.objects.new("AmbientSun", sun_data)
    bpy.context.collection.objects.link(sun_obj)
    sun_data.energy = 1.5
    sun_obj.rotation_euler = (math.radians(45), math.radians(15), math.radians(60))


def create_welding_rig():
    """
    Create TCP Rig, Camera parented to TCP, Molten Weld Pool,
    Electric Arc Light, and Spark Spatter Container.
    """
    scene = bpy.context.scene

    # 1. Tool Center Point (TCP) Empty Rig
    tcp_empty = bpy.data.objects.new("TCP_Rig", None)
    tcp_empty.empty_display_type = 'CROSS'
    tcp_empty.empty_display_size = 0.05
    bpy.context.collection.objects.link(tcp_empty)

    # 2. Camera Parented to TCP (Eye-in-Hand inspection camera)
    cam_data = bpy.data.cameras.new(name='MacroTorchCamera')
    cam_obj = bpy.data.objects.new('MacroTorchCamera', cam_data)
    bpy.context.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    cam_data.type = 'PERSP'
    cam_data.lens = 55.0  # Macro inspection focal length
    cam_obj.parent = tcp_empty
    # Position camera looking directly down at TCP with slight trailing angle
    cam_obj.location = (0.0, -0.06, 0.18)
    cam_obj.rotation_euler = (math.radians(18), 0.0, 0.0)

    # 3. Molten Weld Pool Mesh
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=12, radius=0.016, location=(0, 0, 0))
    weld_pool = bpy.context.active_object
    weld_pool.name = "WeldPool"
    weld_pool.scale = (1.4, 0.9, 0.4)  # Elongated teardrop
    weld_pool.parent = tcp_empty
    weld_pool.location = (0, 0, 0)

    # Weld Pool Emissive Material
    pool_mat = bpy.data.materials.new(name='MoltenPoolMat')
    pool_mat.use_nodes = True
    p_nodes = pool_mat.node_tree.nodes
    p_nodes.clear()
    emit_node = p_nodes.new(type='ShaderNodeEmission')
    emit_node.inputs['Color'].default_value = (1.0, 0.85, 0.55, 1.0)  # Incandescent white-yellow
    emit_node.inputs['Strength'].default_value = 35.0
    p_out = p_nodes.new(type='ShaderNodeOutputMaterial')
    pool_mat.node_tree.links.new(emit_node.outputs['Emission'], p_out.inputs['Surface'])
    weld_pool.data.materials.append(pool_mat)

    # 4. Electric Arc Point Light
    arc_data = bpy.data.lights.new(name="ArcLight", type='POINT')
    arc_obj = bpy.data.objects.new("ArcLight", arc_data)
    bpy.context.collection.objects.link(arc_obj)
    arc_obj.parent = tcp_empty
    arc_obj.location = (0, 0, 0.005)
    arc_data.color = (0.85, 0.92, 1.0)  # Bright blue-white electric arc
    arc_data.energy = 85.0
    arc_data.shadow_soft_size = 0.008

    # 5. Spatter Spark Template & Spark Container Group
    spark_mat = bpy.data.materials.new(name='SparkMat')
    spark_mat.use_nodes = True
    s_nodes = spark_mat.node_tree.nodes
    s_nodes.clear()
    s_emit = s_nodes.new(type='ShaderNodeEmission')
    s_emit.inputs['Color'].default_value = (1.0, 0.65, 0.15, 1.0)  # Incandescent orange-red
    s_emit.inputs['Strength'].default_value = 50.0
    s_out = s_nodes.new(type='ShaderNodeOutputMaterial')
    spark_mat.node_tree.links.new(s_emit.outputs['Emission'], s_out.inputs['Surface'])

    return {
        'tcp': tcp_empty,
        'cam': cam_obj,
        'pool': weld_pool,
        'pool_emit': emit_node,
        'arc': arc_obj,
        'arc_data': arc_data,
        'spark_mat': spark_mat
    }


def render_video_sequence(seq_id, seq_type, anomaly_params, rig, out_dir):
    """
    Renders a 30-frame sequence (1.0 second) with dynamic TCP movement,
    arc flickering, procedural spatter, and optional anomaly injection.
    """
    seq_dir = out_dir / seq_id
    seq_dir.mkdir(parents=True, exist_ok=True)

    tcp = rig['tcp']
    arc_data = rig['arc_data']
    pool_emit = rig['pool_emit']
    spark_mat = rig['spark_mat']

    # Seam vector
    p_start = list(SEAM_START)
    p_end = list(SEAM_END)

    # Active sparks list: [{'obj': obj, 'vel': [vx, vy, vz], 'life': int, 'max_life': int}]
    active_sparks = []

    metadata = {
        'sequence_id': seq_id,
        'sequence_type': seq_type,
        'anomaly_type': anomaly_params.get('type', 'NONE'),
        'anomaly_start_frame': anomaly_params.get('start_frame', -1),
        'anomaly_end_frame': anomaly_params.get('end_frame', -1),
        'frames_count': FRAMES_PER_SEQ,
        'fps': FPS,
        'frames': []
    }

    base_arc_energy = 85.0
    base_pool_strength = 35.0

    print(f"\n[AutoTwin Video] Rendering Sequence '{seq_id}' [{seq_type} - {anomaly_params.get('type', 'NONE')}]")

    for f in range(FRAMES_PER_SEQ):
        t_norm = f / float(max(1, FRAMES_PER_SEQ - 1))

        # Check if current frame is in anomaly window
        is_anomaly_frame = False
        if seq_type == 'ANOMALOUS':
            a_start = anomaly_params.get('start_frame', 10)
            a_end = anomaly_params.get('end_frame', 22)
            if a_start <= f <= a_end:
                is_anomaly_frame = True

        # -----------------------------------------------------
        # 1. Update TCP Position along seam (with optional Jerk anomaly)
        # -----------------------------------------------------
        cur_t = t_norm
        if is_anomaly_frame and anomaly_params.get('type') == 'VELOCITY_JERK':
            # Jerk / stutter: robot decelerates violently and shifts laterally
            jerk_offset = math.sin((f - a_start) * 1.5) * 0.018
            cur_x = p_start[0] + cur_t * (p_end[0] - p_start[0])
            cur_y = p_start[1] + jerk_offset
            cur_z = p_start[2]
        else:
            cur_x = p_start[0] + cur_t * (p_end[0] - p_start[0])
            cur_y = p_start[1] + cur_t * (p_end[1] - p_start[1])
            cur_z = p_start[2] + cur_t * (p_end[2] - p_start[2])

        tcp.location = (cur_x, cur_y, cur_z)

        # -----------------------------------------------------
        # 2. Arc Light & Weld Pool Dynamics
        # -----------------------------------------------------
        if is_anomaly_frame and anomaly_params.get('type') == 'ARC_DROP':
            # Power drop: arc extinguishes to 5%, weld pool cools
            arc_data.energy = base_arc_energy * random.uniform(0.02, 0.08)
            pool_emit.inputs['Strength'].default_value = base_pool_strength * 0.15
            pool_emit.inputs['Color'].default_value = (0.5, 0.15, 0.05, 1.0)  # Dark dull red
        else:
            # Normal realistic arc flicker (+-20%)
            flicker = random.uniform(0.80, 1.25)
            arc_data.energy = base_arc_energy * flicker
            pool_emit.inputs['Strength'].default_value = base_pool_strength * flicker
            pool_emit.inputs['Color'].default_value = (1.0, 0.85, 0.55, 1.0)

        # -----------------------------------------------------
        # 3. Procedural Spatter Particle Simulation
        # -----------------------------------------------------
        # Clean up dead sparks from previous frame
        surviving_sparks = []
        for sp in active_sparks:
            sp['life'] += 1
            if sp['life'] < sp['max_life']:
                # Update physics: velocity + gravity
                loc = sp['obj'].location
                loc.x += sp['vel'][0]
                loc.y += sp['vel'][1]
                loc.z += sp['vel'][2]
                sp['vel'][2] -= 0.003  # Gravity
                surviving_sparks.append(sp)
            else:
                bpy.data.objects.remove(sp['obj'], do_unlink=True)
        active_sparks = surviving_sparks

        # Spawn new sparks for current frame
        num_new_sparks = 0
        if is_anomaly_frame and anomaly_params.get('type') == 'SPATTER_BURST':
            # Extreme violent spatter explosion
            num_new_sparks = random.randint(18, 35)
            arc_data.energy *= 1.6  # Flash burst
        else:
            # Nominal occasional micro-spark
            num_new_sparks = random.randint(0, 2)

        for _ in range(num_new_sparks):
            bpy.ops.mesh.primitive_uv_sphere_add(
                segments=6, ring_count=4,
                radius=random.uniform(0.0015, 0.0035),
                location=(cur_x, cur_y, cur_z + 0.005)
            )
            spark_obj = bpy.context.active_object
            spark_obj.name = f"Spark_{f}_{len(active_sparks)}"
            spark_obj.data.materials.append(spark_mat)

            # Random trajectory radiating away from TCP
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.015, 0.045) if (is_anomaly_frame and anomaly_params.get('type') == 'SPATTER_BURST') else random.uniform(0.005, 0.015)
            elevation = random.uniform(0.01, 0.035)

            active_sparks.append({
                'obj': spark_obj,
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed, elevation],
                'life': 0,
                'max_life': random.randint(4, 10)
            })

        # -----------------------------------------------------
        # 4. Render Frame
        # -----------------------------------------------------
        frame_filename = f"frame_{f:03d}.png"
        frame_path = seq_dir / frame_filename
        bpy.context.scene.render.filepath = str(frame_path)

        t_render = time.time()
        bpy.ops.render.render(write_still=True)
        render_duration = time.time() - t_render

        frame_meta = {
            'frame_index': f,
            'filename': frame_filename,
            'tcp_position': [round(cur_x, 4), round(cur_y, 4), round(cur_z, 4)],
            'arc_energy': round(arc_data.energy, 2),
            'spatter_particle_count': len(active_sparks),
            'is_anomaly': is_anomaly_frame,
            'render_time_sec': round(render_duration, 3)
        }
        metadata['frames'].append(frame_meta)

        status_tag = "ANOMALY" if is_anomaly_frame else "NORMAL"
        print(f"  Frame {f:02d}/{FRAMES_PER_SEQ - 1:02d} | {status_tag} | Sparks: {len(active_sparks):02d} | Arc: {arc_data.energy:5.1f}W | {render_duration:.2f}s")

    # Clean up any remaining sparks
    for sp in active_sparks:
        bpy.data.objects.remove(sp['obj'], do_unlink=True)

    # Save sequence metadata
    with open(seq_dir / "meta.json", "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

    return metadata


def main():
    print("=" * 70)
    print("AutoTwin-AI v2.0: Spatiotemporal Video Dataset Generator")
    print(f"Target FPS: {FPS} | Frames per Sequence: {FRAMES_PER_SEQ} (1.0s video)")
    print(f"Normal Sequences: {NUM_NORMAL_SEQS} | Spatter Burst: {NUM_SPATTER_SEQS} | Arc Drop: {NUM_ARCDROP_SEQS} | Velocity Jerk: {NUM_JERK_SEQS}")
    print("=" * 70)

    clean_scene()
    setup_render_engine()
    chassis = import_and_setup_chassis()
    create_world_and_lighting()
    rig = create_welding_rig()

    normal_dir = DATASET_DIR / "normal"
    anomalous_dir = DATASET_DIR / "anomalous"
    normal_dir.mkdir(parents=True, exist_ok=True)
    anomalous_dir.mkdir(parents=True, exist_ok=True)

    all_manifest = []

    # 1. Generate Normal Sequences
    for i in range(NUM_NORMAL_SEQS):
        seq_id = f"normal_seq_{i:03d}"
        meta = render_video_sequence(seq_id, 'NORMAL', {}, rig, normal_dir)
        all_manifest.append(meta)

    # 2. Generate Spatter Burst Anomalies
    for i in range(NUM_SPATTER_SEQS):
        seq_id = f"anom_spatter_{i:03d}"
        params = {'type': 'SPATTER_BURST', 'start_frame': 10, 'end_frame': 20}
        meta = render_video_sequence(seq_id, 'ANOMALOUS', params, rig, anomalous_dir)
        all_manifest.append(meta)

    # 3. Generate Arc Drop Anomalies
    for i in range(NUM_ARCDROP_SEQS):
        seq_id = f"anom_arcdrop_{i:03d}"
        params = {'type': 'ARC_DROP', 'start_frame': 12, 'end_frame': 22}
        meta = render_video_sequence(seq_id, 'ANOMALOUS', params, rig, anomalous_dir)
        all_manifest.append(meta)

    # 4. Generate Velocity Jerk Anomalies
    for i in range(NUM_JERK_SEQS):
        seq_id = f"anom_jerk_{i:03d}"
        params = {'type': 'VELOCITY_JERK', 'start_frame': 8, 'end_frame': 18}
        meta = render_video_sequence(seq_id, 'ANOMALOUS', params, rig, anomalous_dir)
        all_manifest.append(meta)

    # Save Global Manifest
    manifest_path = DATASET_DIR / "sequence_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(all_manifest, fp, indent=2)

    print("\n" + "=" * 70)
    print(f"[SUCCESS] Generated {len(all_manifest)} video sequences!")
    print(f"Manifest saved to: {manifest_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
