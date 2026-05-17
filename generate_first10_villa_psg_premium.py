#!/usr/bin/env python3
"""
Premium first-10-seconds football scene generator for Blender 4.x/5.x.

Run with:
    blender --background --python generate_first10_villa_psg_premium.py

This remains a procedural authoring scene, not a photoreal asset pack. The goal
is to replace the placeholder look with a stronger premium baseline while
keeping every major system data-driven for later tracking integration.
"""

import math
import os
from mathutils import Vector

import bpy


FPS = 25
START_FRAME = 1
END_FRAME = 250
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
OUTPUT_FILE = "first10_villa_psg_premium.blend"
# Point to your active Aston Villa vs PSG video in the Desktop folder
VIDEO_PATH = "/Users/rudra/Desktop/untitled folder 2/aston_villa_psg_clip1_annotated.mp4"
TACTICAL_OVERLAYS = True


VILLA_STARTS = [
    (-38, -10, 0),
    (-34, 10, 0),
    (-28, -22, 0),
    (-26, 0, 0),
    (-25, 21, 0),
    (-16, -16, 0),
    (-14, 5, 0),
    (-12, 23, 0),
    (-4, -8, 0),
    (-2, 12, 0),
    (8, 0, 0),
]
VILLA_ENDS = [
    (-40, -8, 0),
    (-36, 12, 0),
    (-31, -20, 0),
    (-29, 1, 0),
    (-27, 19, 0),
    (-18, -13, 0),
    (-17, 7, 0),
    (-15, 21, 0),
    (-8, -6, 0),
    (-6, 10, 0),
    (2, 1, 0),
]
PSG_STARTS = [
    (30, 0, 0),
    (24, -18, 0),
    (24, 18, 0),
    (16, -9, 0),
    (14, 10, 0),
    (8, -24, 0),
    (7, 23, 0),
    (2, -4, 0),
    (0, 13, 0),
    (-7, -15, 0),
    (-10, 5, 0),
]
PSG_ENDS = [
    (18, -3, 0),
    (14, -20, 0),
    (12, 17, 0),
    (4, -11, 0),
    (2, 8, 0),
    (-2, -25, 0),
    (-3, 22, 0),
    (-8, -7, 0),
    (-10, 12, 0),
    (-16, -14, 0),
    (-19, 2, 0),
]
BALL_KEYFRAMES = [
    (1, (16, -9, 0.22)),
    (55, (7, -5, 0.26)),
    (95, (0, 8, 0.24)),
    (145, (-7, 12, 0.28)),
    (200, (-14, 5, 0.25)),
    (250, (-19, 2, 0.23)),
]


def ensure_collection(name, parent=None):
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        if parent is None:
            bpy.context.scene.collection.children.link(collection)
        else:
            parent.children.link(collection)
    return collection


def move_to_collection(obj, collection):
    for existing in list(obj.users_collection):
        existing.objects.unlink(obj)
    collection.objects.link(obj)
    return obj


def make_material(
    name,
    color,
    roughness=0.5,
    metallic=0.0,
    alpha=1.0,
    emission=None,
    emission_strength=0.0,
):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Metallic"].default_value = metallic
        bsdf.inputs["Alpha"].default_value = alpha
        if emission is not None:
            bsdf.inputs["Emission Color"].default_value = emission
            bsdf.inputs["Emission Strength"].default_value = emission_strength
    if alpha < 1.0:
        material.blend_method = "BLEND"
        material.use_screen_refraction = True
        material.show_transparent_back = True
    return material


def shade_smooth(obj):
    if hasattr(obj.data, "polygons"):
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
    return obj


def add_cube(name, loc, scale, material, collection):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if material:
        obj.data.materials.append(material)
    return move_to_collection(obj, collection)


def add_cylinder(name, loc, radius, depth, material, collection, vertices=48, rotation=(0, 0, 0)):
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=radius,
        depth=depth,
        location=loc,
        rotation=rotation,
    )
    obj = bpy.context.object
    obj.name = name
    if material:
        obj.data.materials.append(material)
    shade_smooth(obj)
    return move_to_collection(obj, collection)


def add_uv_sphere(name, loc, radius, material, collection, segments=32, rings=16):
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=segments,
        ring_count=rings,
        radius=radius,
        location=loc,
    )
    obj = bpy.context.object
    obj.name = name
    if material:
        obj.data.materials.append(material)
    shade_smooth(obj)
    return move_to_collection(obj, collection)


def add_torus(name, loc, major_radius, minor_radius, material, collection, major_segments=128, minor_segments=12):
    bpy.ops.mesh.primitive_torus_add(
        major_segments=major_segments,
        minor_segments=minor_segments,
        major_radius=major_radius,
        minor_radius=minor_radius,
        location=loc,
    )
    obj = bpy.context.object
    obj.name = name
    if material:
        obj.data.materials.append(material)
    shade_smooth(obj)
    return move_to_collection(obj, collection)


def add_curve_line(name, points, bevel_depth, material, collection):
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 12
    curve.bevel_depth = bevel_depth
    curve.resolution_v = 2
    spline = curve.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for point, loc in zip(spline.points, points):
        point.co = (loc[0], loc[1], loc[2], 1)
    obj = bpy.data.objects.new(name, curve)
    if material:
        curve.materials.append(material)
    bpy.context.collection.objects.link(obj)
    return move_to_collection(obj, collection)


def add_text(name, text, loc, size, material, collection, align="CENTER"):
    curve = bpy.data.curves.new(f"{name}_curve", "FONT")
    curve.body = text
    curve.size = size
    curve.align_x = align
    curve.align_y = "CENTER"
    obj = bpy.data.objects.new(name, curve)
    obj.location = loc
    obj.rotation_euler = (math.radians(70), 0, 0)
    if material:
        curve.materials.append(material)
    bpy.context.collection.objects.link(obj)
    return move_to_collection(obj, collection)


def look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def insert_location_key(obj, frame, loc):
    obj.location = loc
    obj.keyframe_insert(data_path="location", frame=frame)


def insert_rotation_key(obj, frame, rotation):
    obj.rotation_euler = rotation
    obj.keyframe_insert(data_path="rotation_euler", frame=frame)


def set_interpolation():
    # Smooth all keyframe interpolation (handling legacy & Blender 5.0+ slots API)
    for obj in bpy.data.objects:
        if obj.animation_data and obj.animation_data.action:
            action = obj.animation_data.action
            fcurves = []
            if hasattr(action, "fcurves"):
                fcurves = list(action.fcurves)
            else:
                # Blender 5.0+ slot actions
                try:
                    from bpy_extras import anim_utils
                    slot = obj.animation_data.action_slot
                    if slot:
                        cb = anim_utils.action_get_channelbag_for_slot(action, slot)
                        if cb and hasattr(cb, "fcurves"):
                            fcurves = list(cb.fcurves)
                except Exception:
                    pass
                if not fcurves:
                    # Generic fallback scanning all layers/strips/channelbags
                    for layer in getattr(action, "layers", []):
                        for strip in getattr(layer, "strips", []):
                            cb = getattr(strip, "channelbag", None)
                            if cb and hasattr(cb, "fcurves"):
                                fcurves.extend(cb.fcurves)
            for fc in fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = "BEZIER"


def configure_render(scene):
    scene.render.fps = FPS
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 128
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.preview_samples = 32
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = 0.35
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.world.color = (0.01, 0.012, 0.02)


def build_pitch(materials, collections):
    pitch = collections["pitch"]
    grass_dark = materials["grass_dark"]
    grass_light = materials["grass_light"]
    grass_variation = materials["grass_variation"]
    paint = materials["paint"]

    add_cube(
        "UEFA pitch base",
        (0, 0, -0.035),
        (PITCH_LENGTH + 8, PITCH_WIDTH + 8, 0.07),
        materials["earth"],
        pitch,
    )
    add_cube(
        "UEFA grass pitch",
        (0, 0, 0),
        (PITCH_LENGTH, PITCH_WIDTH, 0.035),
        grass_dark,
        pitch,
    )

    stripe_width = PITCH_LENGTH / 14
    for i in range(14):
        x = -PITCH_LENGTH / 2 + (i + 0.5) * stripe_width
        stripe_material = grass_light if i % 2 else grass_dark
        add_cube(
            f"Mowing stripe {i + 1:02d}",
            (x, 0, 0.02),
            (stripe_width, PITCH_WIDTH, 0.01),
            stripe_material,
            pitch,
        )

    patch_specs = [
        ((-35, -18, 0.026), (17, 10, 0.006)),
        ((-5, 12, 0.026), (24, 12, 0.006)),
        ((28, -11, 0.026), (20, 14, 0.006)),
        ((12, 22, 0.026), (14, 7, 0.006)),
    ]
    for idx, (loc, scale) in enumerate(patch_specs, start=1):
        add_cube(f"Grass wear variation {idx}", loc, scale, grass_variation, pitch)

    line_z = 0.045
    line_h = 0.018
    add_cube("Halfway line", (0, 0, line_z), (0.14, PITCH_WIDTH, line_h), paint, pitch)
    add_cube("Near touchline", (0, -PITCH_WIDTH / 2, line_z), (PITCH_LENGTH, 0.12, line_h), paint, pitch)
    add_cube("Far touchline", (0, PITCH_WIDTH / 2, line_z), (PITCH_LENGTH, 0.12, line_h), paint, pitch)
    add_cube("Left goal line", (-PITCH_LENGTH / 2, 0, line_z), (0.12, PITCH_WIDTH, line_h), paint, pitch)
    add_cube("Right goal line", (PITCH_LENGTH / 2, 0, line_z), (0.12, PITCH_WIDTH, line_h), paint, pitch)
    add_torus("Center circle", (0, 0, 0.055), 9.15, 0.06, paint, pitch)
    add_uv_sphere("Center spot", (0, 0, 0.065), 0.17, paint, pitch, segments=24, rings=12)

    for side, x in [("left", -PITCH_LENGTH / 2), ("right", PITCH_LENGTH / 2)]:
        sx = -1 if side == "left" else 1
        add_cube(f"{side} penalty box top", (x + sx * 8.25, 20.16, line_z), (16.5, 0.12, line_h), paint, pitch)
        add_cube(f"{side} penalty box bottom", (x + sx * 8.25, -20.16, line_z), (16.5, 0.12, line_h), paint, pitch)
        add_cube(f"{side} penalty box front", (x + sx * 16.5, 0, line_z), (0.12, 40.32, line_h), paint, pitch)
        add_cube(f"{side} six yard top", (x + sx * 2.75, 9.16, line_z), (5.5, 0.12, line_h), paint, pitch)
        add_cube(f"{side} six yard bottom", (x + sx * 2.75, -9.16, line_z), (5.5, 0.12, line_h), paint, pitch)
        add_cube(f"{side} six yard front", (x + sx * 5.5, 0, line_z), (0.12, 18.32, line_h), paint, pitch)
        add_uv_sphere(f"{side} penalty spot", (x + sx * 11, 0, 0.065), 0.15, paint, pitch, segments=24, rings=12)


def build_goals(materials, collections):
    goal_collection = collections["goals"]
    white = materials["paint"]
    net = materials["net"]

    for side, x in [("left", -PITCH_LENGTH / 2 - 1.1), ("right", PITCH_LENGTH / 2 + 1.1)]:
        sign = -1 if side == "left" else 1
        add_cylinder(f"{side} goal left post", (x, -3.66, 1.22), 0.07, 2.44, white, goal_collection, 32)
        add_cylinder(f"{side} goal right post", (x, 3.66, 1.22), 0.07, 2.44, white, goal_collection, 32)
        crossbar = add_cylinder(
            f"{side} goal crossbar",
            (x, 0, 2.44),
            0.07,
            7.32,
            white,
            goal_collection,
            32,
            rotation=(math.radians(90), 0, 0),
        )
        crossbar.rotation_euler[1] = math.radians(90)
        add_cube(f"{side} goal back net", (x + sign * 1.45, 0, 1.18), (0.04, 7.32, 2.25), net, goal_collection)
        add_cube(f"{side} goal roof net", (x + sign * 0.72, 0, 2.46), (1.45, 7.32, 0.04), net, goal_collection)
        for idx, y in enumerate([-2.44, -1.22, 0, 1.22, 2.44], start=1):
            add_curve_line(
                f"{side} net vertical guide {idx}",
                [(x, y, 0.06), (x + sign * 1.45, y, 0.06), (x + sign * 1.45, y, 2.38)],
                0.012,
                net,
                goal_collection,
            )


def build_stadium(materials, collections):
    stadium = collections["stadium"]
    crowd = collections["crowd"]
    lights = collections["lights"]

    metal = materials["metal"]
    concrete = materials["concrete"]
    crowd_a = materials["crowd_a"]
    crowd_b = materials["crowd_b"]
    crowd_c = materials["crowd_c"]
    led = materials["led"]
    bench = materials["bench"]
    canopy = materials["canopy"]
    lamp = materials["lamp"]

    for y, name in [(-44, "near"), (44, "far")]:
        direction = 1 if y > 0 else -1
        for tier in range(5):
            seat_y = y + direction * tier * 2.0
            add_cube(
                f"{name} bowl tier {tier + 1}",
                (0, seat_y, 1.0 + tier * 1.1),
                (120 - tier * 3.0, 2.8, 0.85),
                concrete,
                stadium,
            )
            crowd_mat = [crowd_a, crowd_b, crowd_c][tier % 3]
            add_cube(
                f"{name} crowd band {tier + 1}",
                (0, seat_y - direction * 0.25, 1.65 + tier * 1.1),
                (116 - tier * 3.0, 0.18, 0.7),
                crowd_mat,
                crowd,
            )

    for x, name in [(-63, "left"), (63, "right")]:
        direction = 1 if x > 0 else -1
        for tier in range(4):
            seat_x = x + direction * tier * 1.8
            add_cube(
                f"{name} end tier {tier + 1}",
                (seat_x, 0, 1.0 + tier * 1.05),
                (2.8, 78 - tier * 3.0, 0.85),
                concrete,
                stadium,
            )

    add_cube("Near team bench", (-18, -38.0, 0.65), (13, 1.4, 1.2), bench, stadium)
    add_cube("Far team bench", (18, 38.0, 0.65), (13, 1.4, 1.2), bench, stadium)
    add_cube("Players tunnel roof", (0, -41.2, 1.6), (8.5, 4.4, 0.2), canopy, stadium)
    add_cube("Players tunnel body", (0, -41.2, 0.9), (8.5, 4.4, 1.5), metal, stadium)

    add_cube("LED near sideline", (0, -36.3, 0.72), (108, 0.18, 1.15), led, stadium)
    add_cube("LED far sideline", (0, 36.3, 0.72), (108, 0.18, 1.15), led, stadium)
    add_cube("LED left end", (-54.6, 0, 0.72), (0.18, 69, 1.15), led, stadium)
    add_cube("LED right end", (54.6, 0, 0.72), (0.18, 69, 1.15), led, stadium)

    for x in [-42, 0, 42]:
        add_cube(f"Near roof truss {x}", (x, -48.5, 10.5), (20, 0.45, 0.45), metal, stadium)
        add_cube(f"Far roof truss {x}", (x, 48.5, 10.5), (20, 0.45, 0.45), metal, stadium)
    add_cube("Near canopy", (0, -48.5, 11.0), (124, 6.2, 0.3), canopy, stadium)
    add_cube("Far canopy", (0, 48.5, 11.0), (124, 6.2, 0.3), canopy, stadium)

    for i, (x, y) in enumerate([(-48, -49), (0, -50), (48, -49), (-48, 49), (0, 50), (48, 49)], start=1):
        add_cube(f"Floodlight tower {i}", (x, y, 12.5), (0.42, 0.42, 25), metal, lights)
        add_cube(f"Floodlight bank {i}", (x, y, 25.7), (8.5, 0.5, 2.4), lamp, lights)
        for j, dz in enumerate([-0.65, 0, 0.65], start=1):
            bpy.ops.object.light_add(type="AREA", location=(x, y, 24.7 + dz))
            light = bpy.context.object
            light.name = f"Area floodlight {i}-{j}"
            light.data.energy = 850 if abs(x) > 1 else 700
            light.data.size = 9
            light.data.use_shadow = True
            look_at(light, (0, 0, 0))
            move_to_collection(light, lights)

    add_cube("Atmospheric haze volume", (0, 0, 12), (126, 104, 24), materials["haze"], stadium)


def animate_limb(obj, phase=1.0):
    keys = [
        (1, math.radians(18 * phase)),
        (63, math.radians(-20 * phase)),
        (126, math.radians(18 * phase)),
        (188, math.radians(-20 * phase)),
        (250, math.radians(18 * phase)),
    ]
    for frame, angle in keys:
        obj.rotation_euler = (angle, 0, 0)
        obj.keyframe_insert(data_path="rotation_euler", frame=frame)


def build_player(name, team, number, start, end, materials, collection):
    root = bpy.data.objects.new(name, None)
    root.empty_display_type = "SPHERE"
    root.empty_display_size = 0.35
    root.location = start
    bpy.context.collection.objects.link(root)
    move_to_collection(root, collection)

    direction = Vector((end[0] - start[0], end[1] - start[1], 0))
    if direction.length > 0:
        root.rotation_euler[2] = math.atan2(direction.y, direction.x) - math.radians(90)

    team_main = materials["villa_main"] if team == "Aston Villa" else materials["psg_main"]
    team_trim = materials["villa_trim"] if team == "Aston Villa" else materials["psg_trim"]

    shadow = add_uv_sphere(f"{name}_contact_shadow", (0, 0, 0.045), 0.34, materials["shadow"], collection, 24, 12)
    shadow.scale = (1.1, 0.75, 0.08)
    shadow.parent = root

    torso = add_uv_sphere(f"{name}_torso", (0, 0, 1.12), 0.34, team_main, collection, 24, 12)
    torso.scale = (0.78, 0.52, 1.2)
    torso.parent = root

    chest_panel = add_cube(f"{name}_shirt_panel", (0, -0.29, 1.14), (0.16, 0.03, 0.56), team_trim, collection)
    chest_panel.parent = root

    shorts = add_uv_sphere(f"{name}_shorts", (0, 0, 0.73), 0.27, team_main, collection, 20, 10)
    shorts.scale = (0.95, 0.72, 0.6)
    shorts.parent = root

    head = add_uv_sphere(f"{name}_head", (0, 0, 1.68), 0.19, materials["skin"], collection, 24, 12)
    head.parent = root

    for side in (-1, 1):
        upper_arm = add_cylinder(
            f"{name}_upper_arm_{side}",
            (0.29 * side, 0, 1.18),
            0.065,
            0.42,
            team_trim,
            collection,
            20,
            rotation=(math.radians(8 * side), 0, 0),
        )
        upper_arm.parent = root
        animate_limb(upper_arm, phase=side)

        forearm = add_cylinder(
            f"{name}_forearm_{side}",
            (0.32 * side, -0.02, 0.89),
            0.055,
            0.34,
            materials["skin"],
            collection,
            20,
        )
        forearm.parent = root
        animate_limb(forearm, phase=side)

        thigh = add_cylinder(
            f"{name}_thigh_{side}",
            (0.13 * side, 0, 0.49),
            0.095,
            0.44,
            materials["skin"],
            collection,
            20,
        )
        thigh.parent = root
        animate_limb(thigh, phase=-side)

        sock = add_cylinder(
            f"{name}_sock_{side}",
            (0.13 * side, 0, 0.22),
            0.075,
            0.30,
            team_trim,
            collection,
            20,
        )
        sock.parent = root
        animate_limb(sock, phase=-side)

        boot = add_cube(
            f"{name}_boot_{side}",
            (0.13 * side, -0.08, 0.06),
            (0.16, 0.34, 0.11),
            materials["boots"],
            collection,
        )
        boot.parent = root

    number_curve = bpy.data.curves.new(f"{name}_number_curve", "FONT")
    number_curve.body = str(number)
    number_curve.align_x = "CENTER"
    number_curve.align_y = "CENTER"
    number_curve.size = 0.24
    number = bpy.data.objects.new(f"{name}_number", number_curve)
    number.location = (0, -0.335, 1.18)
    number.rotation_euler = (math.radians(90), 0, 0)
    number.parent = root
    number_curve.materials.append(materials["paint"])
    bpy.context.collection.objects.link(number)
    move_to_collection(number, collection)

    insert_location_key(root, START_FRAME, start)
    insert_location_key(root, END_FRAME, end)
    root["team"] = team
    root["shirt_number"] = str(number_curve.body)
    return root


def build_players(materials, collections):
    players = collections["players"]
    for idx, (start, end) in enumerate(zip(VILLA_STARTS, VILLA_ENDS), start=1):
        build_player(f"AVL_Player_{idx:02d}", "Aston Villa", idx, start, end, materials, players)
    for idx, (start, end) in enumerate(zip(PSG_STARTS, PSG_ENDS), start=1):
        build_player(f"PSG_Player_{idx:02d}", "PSG", idx, start, end, materials, players)


def build_ball(materials, collections):
    ball_collection = collections["ball"]
    ball = add_uv_sphere("Match ball", BALL_KEYFRAMES[0][1], 0.22, materials["ball"], ball_collection, 32, 16)
    for frame, loc in BALL_KEYFRAMES:
        insert_location_key(ball, frame, loc)
    insert_rotation_key(ball, START_FRAME, (0, 0, 0))
    insert_rotation_key(ball, END_FRAME, (math.radians(720), math.radians(240), math.radians(120)))
    return ball


def build_overlays(materials, collections):
    overlays = collections["overlays"]
    if not TACTICAL_OVERLAYS:
        overlays.hide_render = True
        overlays.hide_viewport = True
        return

    glass = materials["overlay_blue"]
    accent = materials["overlay_green"]
    warning = materials["overlay_red"]
    white = materials["paint"]

    add_torus("Possession halo", (16, -9, 0.08), 0.72, 0.045, accent, overlays, 96, 12)
    add_text("Carrier label", "BALL CARRIER", (16, -9, 0.8), 0.28, white, overlays)
    add_text("Pressing title", "FIRST 10 SECONDS", (0, -40.8, 3.2), 0.72, white, overlays)
    add_text("Pressing subtitle", "BROADCAST RECONSTRUCTION / PLACEHOLDER TRACKS", (0, -40.8, 2.45), 0.34, glass, overlays)

    points = [(x, y, 0.09) for _, (x, y, _) in BALL_KEYFRAMES]
    add_curve_line("Trajectory ribbon", points, 0.06, glass, overlays)

    add_curve_line(
        "Passing lane sample",
        [(-7, 12, 0.09), (-12, 8, 0.09), (-18, 3, 0.09)],
        0.05,
        warning,
        overlays,
    )


def build_reference_screen(materials, collections):
    screens = collections["screens"]
    screen = add_cube("Reference video screen", (0, -58, 8), (22, 0.12, 12.375), materials["screen_fallback"], screens)
    if VIDEO_PATH and os.path.exists(VIDEO_PATH):
        material = bpy.data.materials.new("Reference video material")
        material.use_nodes = True
        nodes = material.node_tree.nodes
        bsdf = nodes.get("Principled BSDF")
        texture = nodes.new("ShaderNodeTexImage")
        image = bpy.data.images.load(VIDEO_PATH)
        image.source = "MOVIE"
        texture.image = image
        if hasattr(texture, "image_user"):
            texture.image_user.frame_start = START_FRAME
            texture.image_user.frame_duration = END_FRAME
            texture.image_user.use_auto_refresh = True
        material.node_tree.links.new(texture.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Emission Color"].default_value = (1, 1, 1, 1)
        bsdf.inputs["Emission Strength"].default_value = 0.25
        screen.data.materials.clear()
        screen.data.materials.append(material)
    return screen


def build_cameras(collections):
    cameras = collections["cameras"]

    bpy.ops.object.camera_add(location=(20, -56, 25))
    broadcast = bpy.context.object
    broadcast.name = "Camera_Broadcast"
    broadcast.data.lens = 42
    broadcast.data.dof.use_dof = True
    broadcast.data.dof.focus_distance = 57
    broadcast.data.dof.aperture_fstop = 6.3
    move_to_collection(broadcast, cameras)
    camera_path = [
        (1, (20, -56, 25), (10, -4, 0)),
        (80, (8, -56, 25), (0, 4, 0)),
        (160, (-6, -56, 25), (-10, 7, 0)),
        (250, (-18, -56, 24), (-18, 2, 0)),
    ]
    for frame, loc, target in camera_path:
        broadcast.location = loc
        look_at(broadcast, target)
        broadcast.keyframe_insert(data_path="location", frame=frame)
        broadcast.keyframe_insert(data_path="rotation_euler", frame=frame)

    bpy.ops.object.camera_add(location=(0, 0, 82))
    tactical = bpy.context.object
    tactical.name = "Camera_Tactical"
    tactical.data.lens = 24
    look_at(tactical, (0, 0, 0))
    move_to_collection(tactical, cameras)

    bpy.ops.object.camera_add(location=(-8, -18, 2.2))
    player_review = bpy.context.object
    player_review.name = "Camera_Player_Review"
    player_review.data.lens = 28
    look_at(player_review, (-17, 2, 1.2))
    move_to_collection(player_review, cameras)

    return broadcast


def create_materials():
    return {
        "grass_dark": make_material("Grass dark", (0.035, 0.18, 0.045, 1), roughness=0.9),
        "grass_light": make_material("Grass light", (0.05, 0.27, 0.06, 1), roughness=0.92),
        "grass_variation": make_material("Grass variation", (0.08, 0.19, 0.045, 0.24), roughness=1.0, alpha=0.24),
        "earth": make_material("Pitch earth", (0.06, 0.045, 0.025, 1), roughness=1.0),
        "paint": make_material("Pitch paint", (0.95, 0.96, 0.93, 1), roughness=0.35),
        "villa_main": make_material("Aston Villa claret", (0.31, 0.015, 0.08, 1), roughness=0.48),
        "villa_trim": make_material("Aston Villa sky trim", (0.30, 0.68, 0.93, 1), roughness=0.45),
        "psg_main": make_material("PSG navy", (0.01, 0.02, 0.08, 1), roughness=0.45),
        "psg_trim": make_material("PSG red trim", (0.72, 0.03, 0.06, 1), roughness=0.45),
        "skin": make_material("Skin", (0.64, 0.43, 0.31, 1), roughness=0.5),
        "boots": make_material("Boots", (0.01, 0.01, 0.012, 1), roughness=0.3),
        "ball": make_material("Match ball yellow", (0.98, 0.76, 0.08, 1), roughness=0.28),
        "shadow": make_material("Contact shadow", (0.01, 0.01, 0.01, 0.32), roughness=1.0, alpha=0.32),
        "metal": make_material("Dark stadium metal", (0.05, 0.055, 0.06, 1), roughness=0.32, metallic=0.42),
        "concrete": make_material("Concrete seating", (0.17, 0.18, 0.20, 1), roughness=0.82),
        "crowd_a": make_material("Crowd deep", (0.10, 0.08, 0.10, 1), roughness=0.8),
        "crowd_b": make_material("Crowd mid", (0.18, 0.12, 0.13, 1), roughness=0.8),
        "crowd_c": make_material("Crowd cool", (0.10, 0.14, 0.18, 1), roughness=0.8),
        "bench": make_material("Bench shell", (0.08, 0.11, 0.14, 1), roughness=0.4, metallic=0.1),
        "canopy": make_material("Canopy", (0.08, 0.09, 0.11, 1), roughness=0.35, metallic=0.15),
        "led": make_material(
            "LED boards",
            (0.10, 0.42, 0.95, 1),
            roughness=0.18,
            emission=(0.10, 0.48, 1.0, 1),
            emission_strength=2.6,
        ),
        "lamp": make_material(
            "Floodlight lamp",
            (1, 1, 1, 1),
            roughness=0.2,
            emission=(1, 1, 1, 1),
            emission_strength=6.0,
        ),
        "net": make_material("Goal net", (0.62, 0.78, 1.0, 0.26), roughness=0.08, alpha=0.26),
        "overlay_blue": make_material(
            "Overlay blue",
            (0.20, 0.64, 1.0, 0.34),
            roughness=0.08,
            alpha=0.34,
            emission=(0.20, 0.64, 1.0, 1),
            emission_strength=0.6,
        ),
        "overlay_green": make_material(
            "Overlay green",
            (0.16, 1.0, 0.48, 0.42),
            roughness=0.08,
            alpha=0.42,
            emission=(0.16, 1.0, 0.48, 1),
            emission_strength=0.75,
        ),
        "overlay_red": make_material(
            "Overlay red",
            (1.0, 0.22, 0.18, 0.42),
            roughness=0.08,
            alpha=0.42,
            emission=(1.0, 0.22, 0.18, 1),
            emission_strength=0.65,
        ),
        "screen_fallback": make_material(
            "Screen fallback",
            (0.05, 0.09, 0.18, 1),
            roughness=0.2,
            emission=(0.08, 0.16, 0.28, 1),
            emission_strength=0.5,
        ),
        "haze": make_material("Atmospheric haze", (0.55, 0.65, 0.85, 0.06), roughness=1.0, alpha=0.06),
    }


def create_collections():
    root = ensure_collection("Athlink Premium Scene")
    names = [
        "pitch",
        "goals",
        "stadium",
        "crowd",
        "lights",
        "players",
        "ball",
        "overlays",
        "screens",
        "cameras",
    ]
    return {name: ensure_collection(name.title(), root) for name in names}


def build_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    scene = bpy.context.scene
    scene.frame_start = START_FRAME
    scene.frame_end = END_FRAME
    configure_render(scene)

    materials = create_materials()
    collections = create_collections()

    build_pitch(materials, collections)
    build_goals(materials, collections)
    build_stadium(materials, collections)
    build_players(materials, collections)
    build_ball(materials, collections)
    build_overlays(materials, collections)
    build_reference_screen(materials, collections)
    broadcast = build_cameras(collections)
    set_interpolation()

    scene.camera = broadcast
    scene["source_segment"] = "Villa vs PSG - First 10 seconds"
    scene["fps"] = FPS
    scene["frame_range"] = f"{START_FRAME}-{END_FRAME}"
    scene["duration"] = "10 seconds"
    scene["players"] = 22
    scene["cameras"] = 3
    scene["generator"] = "premium procedural"

    output_path = os.path.join(bpy.path.abspath("//"), OUTPUT_FILE)
    bpy.ops.wm.save_as_mainfile(filepath=output_path)

    print("=" * 72)
    print("PREMIUM SCENE GENERATED")
    print(f"Output: {output_path}")
    print("Use Camera_Broadcast for the primary render.")
    print("Toggle the Overlays collection if you want a clean base scene.")
    print("=" * 72)


if __name__ == "__main__":
    build_scene()
