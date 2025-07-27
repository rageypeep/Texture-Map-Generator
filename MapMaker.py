import dearpygui.dearpygui as dpg
import numpy as np
from PIL import Image
import cv2
import os
import json
import math

CONFIG_PATH = "mapmaker_config.json"
default_config = {
    "input_dir": "",
    "export_dir": ""
}

# ---- Config Helpers ----

def save_config(input_dir, export_dir):
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump({"input_dir": input_dir, "export_dir": export_dir}, f)
    except Exception as e:
        print(f"Failed to save config: {e}")

def load_config():
    try:
        with open(CONFIG_PATH, "r") as f:
            cfg = json.load(f)
        return cfg.get("input_dir", ""), cfg.get("export_dir", "")
    except Exception:
        return "", ""

# ---- Image Processing ----

img_size = 256
preview_size = 400  # Increased from 256

normal_map = displacement_map = roughness_map = ao_map = metallic_map = None
last_gray = None
input_directory, export_directory = load_config()
if not input_directory:
    input_directory = os.path.expanduser("~")
if not export_directory:
    export_directory = os.path.expanduser("~")

def generate_normal_map(gray, intensity):
    gray = gray.astype(np.float32) / 255.0
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
    normal = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.float32)
    normal[..., 0] = sobelx * intensity
    normal[..., 1] = sobely * intensity
    normal[..., 2] = 1.0
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    normal = (normal / (norm + 1e-8)) * 0.5 + 0.5  # [0,1]
    normal = (normal * 255).astype(np.uint8)
    return Image.fromarray(normal, mode='RGB')

def generate_displacement_map(gray, scale):
    # Apply Gaussian blur to smooth the displacement
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Scale the values
    arr = blurred.astype(np.float32) * scale
    
    # Normalize to 0-255 range with better contrast
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = 255 * (arr - arr_min) / (arr_max - arr_min)
    else:
        arr = np.full_like(arr, 128)  # Mid-gray if no variation
    
    return Image.fromarray(arr.astype(np.uint8), mode='L')

def generate_roughness_map(gray, contrast):
    blur = cv2.GaussianBlur(gray, (15,15), 0)
    rough = cv2.addWeighted(gray, 2*contrast, blur, -1, 128)
    rough = np.clip(rough, 0, 255)
    return Image.fromarray(rough.astype(np.uint8), mode='L')

def generate_ao_map(gray, blur_radius):
    # Convert blur_radius to proper kernel size
    ksize = int(blur_radius)
    if ksize % 2 == 0:
        ksize += 1  # Must be odd
    ksize = max(ksize, 3)
    
    # Create AO by detecting cavities/crevices
    # Method 1: Use morphological operations to find dark areas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    
    # Dilate to find local maxima (peaks)
    dilated = cv2.dilate(gray, kernel)
    
    # AO is the difference between dilated and original
    # Areas that are much darker than their surroundings get more AO
    ao = dilated.astype(np.float32) - gray.astype(np.float32)
    
    # Normalize and invert (darker areas = more occlusion = darker AO)
    ao = np.clip(ao, 0, 255)
    ao = 255 - ao  # Invert so dark crevices become dark AO
    
    # Apply some smoothing
    ao = cv2.GaussianBlur(ao, (5, 5), 0)
    
    return Image.fromarray(ao.astype(np.uint8), mode='L')

def generate_metallic_map(gray, metallic_strength):
    """Generate a metallic map - usually inverse of roughness or based on specific features"""
    # Simple approach: darker areas = more metallic (like metal veins/deposits)
    metallic = 255 - gray  # Invert
    metallic = metallic.astype(np.float32) * metallic_strength
    metallic = np.clip(metallic, 0, 255)
    return Image.fromarray(metallic.astype(np.uint8), mode='L')

def generate_displacement_visualization(gray, scale):
    """Create a height visualization of displacement using fake lighting"""
    # Apply Gaussian blur to smooth the displacement
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Scale the values
    height_map = blurred.astype(np.float32) * scale
    
    # Create fake lighting to show height differences
    # Calculate gradients (slopes)
    grad_x = cv2.Sobel(height_map, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(height_map, cv2.CV_32F, 0, 1, ksize=3)
    
    # Fake light direction (from top-left)
    light_x, light_y = -0.5, -0.5
    
    # Calculate how much light hits each pixel based on slope
    lighting = -(grad_x * light_x + grad_y * light_y)
    
    # Normalize and add to base height
    lighting = lighting / (np.abs(lighting).max() + 1e-8) * 50  # Scale lighting effect
    
    # Combine height and lighting
    result = height_map + lighting + 128  # Add base level
    result = np.clip(result, 0, 255)
    
    return Image.fromarray(result.astype(np.uint8), mode='L')

def generate_pbr_preview():
    """Generate a simple PBR preview using a sphere, square, height viz, or cube"""
    if last_gray is None:
        return np.zeros((preview_size, preview_size, 4), dtype=np.float32)
    
    preview_mode = dpg.get_value("preview_mode")
    
    if preview_mode == "Square":
        return generate_square_preview()
    elif preview_mode == "Height Viz":
        return generate_height_visualization_preview()
    elif preview_mode == "Cube 3D":
        return generate_cube_preview()
    else:
        return generate_sphere_preview()

def generate_square_preview():
    """Generate a flat square preview with lighting"""
    # Create a simple flat square with lighting
    result = np.zeros((preview_size, preview_size, 4), dtype=np.float32)
    
    # Use the center portion of the preview for the square
    margin = preview_size // 8
    square_size = preview_size - 2 * margin
    
    # Sample textures directly
    y_coords = np.linspace(0, img_size-1, square_size).astype(int)
    x_coords = np.linspace(0, img_size-1, square_size).astype(int)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Sample diffuse (base color)
    if last_gray is not None:
        diffuse = last_gray[yy, xx] / 255.0
    else:
        diffuse = np.full((square_size, square_size), 0.5)
    
    # Base normal (pointing up)
    base_normal = np.array([0.0, 0.0, 1.0])
    
    # Sample and apply normal map
    if normal_map is not None:
        normal_rgb = np.array(normal_map.resize((img_size, img_size)))
        normal_sample = normal_rgb[yy, xx] / 255.0 * 2.0 - 1.0
        # For square, we can use the normal map more directly
        perturbed_normals = normal_sample.copy()
        perturbed_normals[:, :, 2] = np.maximum(0.1, perturbed_normals[:, :, 2])  # Ensure Z is positive
        # Normalize
        norm = np.linalg.norm(perturbed_normals, axis=2, keepdims=True)
        perturbed_normals = perturbed_normals / (norm + 1e-8)
    else:
        perturbed_normals = np.tile(base_normal, (square_size, square_size, 1))
    
    # Sample displacement and modify lighting
    displacement_factor = 1.0
    if displacement_map is not None:
        disp_gray = np.array(displacement_map.convert('L').resize((img_size, img_size)))
        displacement = (disp_gray[yy, xx] / 255.0 - 0.5) * 0.3  # Scale displacement effect
        displacement_factor = 1.0 + displacement
    
    # Sample roughness
    if roughness_map is not None:
        roughness_gray = np.array(roughness_map.convert('L').resize((img_size, img_size)))
        roughness = roughness_gray[yy, xx] / 255.0
    else:
        roughness = np.full((square_size, square_size), 0.5)
    
    # Sample AO
    if ao_map is not None:
        ao_gray = np.array(ao_map.convert('L').resize((img_size, img_size)))
        ao = ao_gray[yy, xx] / 255.0
    else:
        ao = np.ones((square_size, square_size))
    
    # Lighting calculation
    light_dir = np.array([0.3, 0.3, 0.8])
    light_dir = light_dir / np.linalg.norm(light_dir)
    
    # Lambertian diffuse
    ndotl = np.maximum(0, np.sum(perturbed_normals * light_dir, axis=2))
    
    # Simple specular
    view_dir = np.array([0, 0, 1])
    half_dir = (light_dir + view_dir) / np.linalg.norm(light_dir + view_dir)
    ndoth = np.maximum(0, np.sum(perturbed_normals * half_dir, axis=2))
    specular_power = 1.0 / (roughness * roughness + 0.01)
    specular = np.power(ndoth, specular_power) * (1.0 - roughness)
    
    # Combine lighting with displacement and AO
    ambient = 0.15
    final_color = diffuse * (ambient + ndotl * 0.7) * ao * displacement_factor + specular * 0.4
    final_color = np.clip(final_color, 0, 1)
    
    # Place the square in the center of the preview
    result[margin:margin+square_size, margin:margin+square_size, 0] = final_color  # R
    result[margin:margin+square_size, margin:margin+square_size, 1] = final_color  # G
    result[margin:margin+square_size, margin:margin+square_size, 2] = final_color  # B
    result[margin:margin+square_size, margin:margin+square_size, 3] = 1.0  # A
    
    return result

def generate_sphere_preview():
    """Generate a sphere preview (original function)"""
    # Create sphere coordinates
    y, x = np.ogrid[-1:1:preview_size*1j, -1:1:preview_size*1j]
    x, y = np.meshgrid(x.flatten(), y.flatten())  # Convert to same shape
    mask = x*x + y*y <= 1
    
    # Calculate sphere normals
    z = np.sqrt(np.maximum(0, 1 - x*x - y*y))
    sphere_normals = np.stack([x, y, z], axis=-1)
    
    # Sample textures at sphere surface
    u = (np.arctan2(y, x) / (2 * np.pi) + 0.5) % 1.0
    v = np.arccos(np.clip(z, -1, 1)) / np.pi
    
    # Convert UV to texture coordinates
    tex_x = (u * (img_size - 1)).astype(int)
    tex_y = (v * (img_size - 1)).astype(int)
    
    # Sample diffuse (base color)
    if last_gray is not None:
        diffuse = last_gray[tex_y, tex_x] / 255.0
    else:
        diffuse = 0.5
    
    # Sample normal map and perturb normals
    if normal_map is not None:
        normal_rgb = np.array(normal_map.resize((img_size, img_size)))
        normal_sample = normal_rgb[tex_y, tex_x] / 255.0 * 2.0 - 1.0
        # Simple normal perturbation
        perturbed_normals = sphere_normals + normal_sample * 0.3
        perturbed_normals = perturbed_normals / (np.linalg.norm(perturbed_normals, axis=-1, keepdims=True) + 1e-8)
    else:
        perturbed_normals = sphere_normals
    
    # Sample roughness
    if roughness_map is not None:
        roughness_gray = np.array(roughness_map.convert('L').resize((img_size, img_size)))
        roughness = roughness_gray[tex_y, tex_x] / 255.0
    else:
        roughness = 0.5
    
    # Sample AO
    if ao_map is not None:
        ao_gray = np.array(ao_map.convert('L').resize((img_size, img_size)))
        ao = ao_gray[tex_y, tex_x] / 255.0
    else:
        ao = 1.0
    
    # Simple lighting calculation
    light_dir = np.array([0.5, 0.5, 0.8])
    light_dir = light_dir / np.linalg.norm(light_dir)
    
    # Lambertian diffuse
    ndotl = np.maximum(0, np.sum(perturbed_normals * light_dir, axis=-1))
    
    # Simple specular
    view_dir = np.array([0, 0, 1])
    half_dir = (light_dir + view_dir) / np.linalg.norm(light_dir + view_dir)
    ndoth = np.maximum(0, np.sum(perturbed_normals * half_dir, axis=-1))
    specular_power = 1.0 / (roughness * roughness + 0.01)
    specular = np.power(ndoth, specular_power) * (1.0 - roughness)
    
    # Combine lighting
    ambient = 0.1
    final_color = diffuse * (ambient + ndotl * 0.8) * ao + specular * 0.3
    final_color = np.clip(final_color, 0, 1)
    
    # Create RGBA output
    result = np.zeros((preview_size, preview_size, 4), dtype=np.float32)
    result[mask, 0] = final_color[mask]  # R
    result[mask, 1] = final_color[mask]  # G
    result[mask, 2] = final_color[mask]  # B
    result[mask, 3] = 1.0  # A
    
    return result

def generate_height_visualization_preview():
    """Generate a preview showing displacement as actual height differences"""
    result = np.zeros((preview_size, preview_size, 4), dtype=np.float32)
    
    # Use the center portion of the preview
    margin = preview_size // 8
    square_size = preview_size - 2 * margin
    
    # Sample displacement map
    if displacement_map is not None:
        disp_gray = np.array(displacement_map.convert('L').resize((square_size, square_size)))
        height = disp_gray.astype(np.float32) / 255.0
    else:
        height = np.full((square_size, square_size), 0.5)
    
    # Create 3D-like visualization using gradients and lighting
    grad_x = np.gradient(height, axis=1)
    grad_y = np.gradient(height, axis=0)
    
    # Multiple light sources for better visualization
    lights = [
        (np.array([-0.6, -0.6, 0.5]), 0.7),  # Main light
        (np.array([0.3, 0.3, 0.8]), 0.3),    # Fill light
    ]
    
    final_lighting = np.zeros_like(height)
    
    for light_dir, intensity in lights:
        light_dir = light_dir / np.linalg.norm(light_dir)
        
        # Calculate surface normals from gradients
        normal_x = -grad_x
        normal_y = -grad_y
        normal_z = np.ones_like(height) * 0.1  # Base normal strength
        
        # Normalize normals
        normal_length = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
        normal_x /= normal_length
        normal_y /= normal_length
        normal_z /= normal_length
        
        # Calculate lighting
        lighting = np.maximum(0, 
            normal_x * light_dir[0] + 
            normal_y * light_dir[1] + 
            normal_z * light_dir[2]
        )
        
        final_lighting += lighting * intensity
    
    # Add ambient lighting and height-based coloring
    ambient = 0.2
    height_color = height * 0.3 + 0.4  # Base color from height
    final_color = height_color * (ambient + final_lighting)
    final_color = np.clip(final_color, 0, 1)
    
    # Place in center of preview
    result[margin:margin+square_size, margin:margin+square_size, 0] = final_color
    result[margin:margin+square_size, margin:margin+square_size, 1] = final_color
    result[margin:margin+square_size, margin:margin+square_size, 2] = final_color
    result[margin:margin+square_size, margin:margin+square_size, 3] = 1.0
    
    return result

def generate_cube_preview():
    """Generate a 3D cube preview with rotation"""
    import time
    
    # Get current time for rotation
    current_time = time.time()
    rotation_speed = dpg.get_value("rotation_speed") if dpg.does_item_exist("rotation_speed") else 0.5
    cube_scale = dpg.get_value("cube_scale") if dpg.does_item_exist("cube_scale") else 0.6
    
    angle_y = (current_time * rotation_speed) % (2 * np.pi)
    angle_x = (current_time * rotation_speed * 0.7) % (2 * np.pi)
    
    # Cube vertices (centered at origin)
    vertices = np.array([
        [-0.5, -0.5, -0.5],  # 0: back-bottom-left
        [ 0.5, -0.5, -0.5],  # 1: back-bottom-right
        [ 0.5,  0.5, -0.5],  # 2: back-top-right
        [-0.5,  0.5, -0.5],  # 3: back-top-left
        [-0.5, -0.5,  0.5],  # 4: front-bottom-left
        [ 0.5, -0.5,  0.5],  # 5: front-bottom-right
        [ 0.5,  0.5,  0.5],  # 6: front-top-right
        [-0.5,  0.5,  0.5],  # 7: front-top-left
    ])
    
    # Cube faces (vertex indices and UV coordinates)
    faces = [
        # Face vertices, normal, UV coords for each vertex
        ([4, 5, 6, 7], [0, 0, 1], [(0,1), (1,1), (1,0), (0,0)]),   # Front (+Z)
        ([1, 0, 3, 2], [0, 0, -1], [(0,1), (1,1), (1,0), (0,0)]),  # Back (-Z)
        ([5, 1, 2, 6], [1, 0, 0], [(0,1), (1,1), (1,0), (0,0)]),   # Right (+X)
        ([0, 4, 7, 3], [-1, 0, 0], [(0,1), (1,1), (1,0), (0,0)]),  # Left (-X)
        ([7, 6, 2, 3], [0, 1, 0], [(0,1), (1,1), (1,0), (0,0)]),   # Top (+Y)
        ([0, 1, 5, 4], [0, -1, 0], [(0,0), (1,0), (1,1), (0,1)]),  # Bottom (-Y)
    ]
    
    # Rotation matrices
    cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
    cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
    
    rot_x = np.array([
        [1, 0, 0],
        [0, cos_x, -sin_x],
        [0, sin_x, cos_x]
    ])
    
    rot_y = np.array([
        [cos_y, 0, sin_y],
        [0, 1, 0],
        [-sin_y, 0, cos_y]
    ])
    
    # Apply rotations
    rotated_vertices = vertices @ rot_x.T @ rot_y.T
    
    # Project to 2D (simple perspective projection)
    camera_distance = 1.8  # Moved camera closer for bigger cube
    projected_vertices = []
    for vertex in rotated_vertices:
        x, y, z = vertex
        # Move camera back
        z += camera_distance
        if z > 0.1:  # Avoid division by zero
            screen_x = x / z
            screen_y = y / z
        else:
            screen_x = screen_y = 0
        projected_vertices.append([screen_x, screen_y, z])
    
    projected_vertices = np.array(projected_vertices)
    
    # Convert to screen coordinates - MUCH BIGGER
    scale = preview_size * cube_scale  # Use the slider value
    center = preview_size // 2
    screen_coords = []
    for vertex in projected_vertices:
        screen_x = int(center + vertex[0] * scale)
        screen_y = int(center - vertex[1] * scale)  # Flip Y
        screen_coords.append([screen_x, screen_y, vertex[2]])
    
    screen_coords = np.array(screen_coords)
    
    # Create output image
    result = np.zeros((preview_size, preview_size, 4), dtype=np.float32)
    depth_buffer = np.full((preview_size, preview_size), float('inf'))
    
    # Light direction
    light_dir = np.array([0.5, 0.5, 0.8])
    light_dir = light_dir / np.linalg.norm(light_dir)
    
    # Rotate light direction with cube for consistent lighting
    rotated_light = light_dir @ rot_x.T @ rot_y.T
    
    # Draw faces (back to front for proper alpha blending)
    face_depths = []
    for i, (face_verts, normal, uvs) in enumerate(faces):
        # Calculate face center depth
        face_center_z = np.mean([rotated_vertices[v][2] for v in face_verts])
        face_depths.append((face_center_z, i))
    
    # Sort faces by depth (furthest first)
    face_depths.sort(key=lambda x: x[0])
    
    for _, face_idx in face_depths:
        face_verts, normal, uvs = faces[face_idx]
        
        # Rotate normal
        rotated_normal = np.array(normal) @ rot_x.T @ rot_y.T
        
        # Back-face culling (skip faces facing away)
        view_dir = np.array([0, 0, -1])
        if np.dot(rotated_normal, view_dir) <= 0:
            continue
        
        # Get screen coordinates for this face
        face_screen_coords = [screen_coords[v] for v in face_verts]
        
        # Calculate lighting
        ndotl = max(0, np.dot(rotated_normal, rotated_light))
        ambient = 0.3
        face_brightness = ambient + ndotl * 0.7
        
        # Rasterize the quad
        rasterize_quad(result, depth_buffer, face_screen_coords, uvs, face_brightness)
    
    return result

def rasterize_quad(image, depth_buffer, screen_coords, uvs, brightness):
    """Rasterize a quad using simple scanline algorithm"""
    # Convert quad to two triangles
    triangles = [
        (screen_coords[0], screen_coords[1], screen_coords[2], uvs[0], uvs[1], uvs[2]),
        (screen_coords[0], screen_coords[2], screen_coords[3], uvs[0], uvs[2], uvs[3])
    ]
    
    for triangle in triangles:
        rasterize_triangle(image, depth_buffer, triangle, brightness)

def rasterize_triangle(image, depth_buffer, triangle_data, brightness):
    """Rasterize a single triangle"""
    (p0, p1, p2, uv0, uv1, uv2) = triangle_data
    
    # Bounding box
    min_x = max(0, int(min(p0[0], p1[0], p2[0])))
    max_x = min(image.shape[1] - 1, int(max(p0[0], p1[0], p2[0])))
    min_y = max(0, int(min(p0[1], p1[1], p2[1])))
    max_y = min(image.shape[0] - 1, int(max(p0[1], p1[1], p2[1])))
    
    # Precompute triangle area for barycentric coordinates
    area = ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]))
    if abs(area) < 1e-6:
        return  # Degenerate triangle
    
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            # Barycentric coordinates
            w0 = ((p1[0] - x) * (p2[1] - y) - (p2[0] - x) * (p1[1] - y)) / area
            w1 = ((p2[0] - x) * (p0[1] - y) - (p0[0] - x) * (p2[1] - y)) / area
            w2 = 1 - w0 - w1
            
            # Check if point is inside triangle
            if w0 >= 0 and w1 >= 0 and w2 >= 0:
                # Interpolate depth
                depth = w0 * p0[2] + w1 * p1[2] + w2 * p2[2]
                
                if depth < depth_buffer[y, x]:
                    depth_buffer[y, x] = depth
                    
                    # Interpolate UV coordinates
                    u = w0 * uv0[0] + w1 * uv1[0] + w2 * uv2[0]
                    v = w0 * uv0[1] + w1 * uv1[1] + w2 * uv2[1]
                    
                    # Sample texture
                    color = sample_texture_at_uv(u, v)
                    
                    # Apply lighting
                    final_color = color * brightness
                    final_color = min(1.0, final_color)
                    
                    # Set pixel
                    image[y, x, 0] = final_color
                    image[y, x, 1] = final_color
                    image[y, x, 2] = final_color
                    image[y, x, 3] = 1.0

def sample_texture_at_uv(u, v):
    """Sample the diffuse texture at UV coordinates"""
    if last_gray is None:
        return 0.5
    
    # Clamp UV coordinates
    u = max(0, min(1, u))
    v = max(0, min(1, v))
    
    # Convert to texture coordinates
    tex_x = int(u * (img_size - 1))
    tex_y = int(v * (img_size - 1))
    
    # Sample diffuse color
    color = last_gray[tex_y, tex_x] / 255.0
    
    # Apply other maps if available
    if normal_map is not None:
        # Normal maps affect lighting (already handled in face brightness)
        pass
    
    if ao_map is not None:
        ao_gray = np.array(ao_map.convert('L').resize((img_size, img_size)))
        ao = ao_gray[tex_y, tex_x] / 255.0
        color *= ao
    
    return color

# ---- GUI Callbacks ----

def load_texture_callback(sender, app_data, user_data):
    global normal_map, displacement_map, roughness_map, ao_map, metallic_map, last_gray, input_directory
    path = app_data['file_path_name']
    input_directory = os.path.dirname(path)
    save_config(input_directory, export_directory)
    dpg.set_value("last_loaded_path", path)
    base_name = os.path.splitext(os.path.basename(path))[0]
    dpg.set_value("export_filename", base_name)
    img = Image.open(path).convert('L').resize((img_size, img_size))
    last_gray = np.array(img)
    update_all_maps_and_previews()

def update_all_maps_and_previews():
    global normal_map, displacement_map, roughness_map, ao_map, metallic_map, last_gray
    norm_int = dpg.get_value("norm_intensity")
    disp_scale = dpg.get_value("disp_scale")
    rough_contrast = dpg.get_value("rough_contrast")
    ao_blur = dpg.get_value("ao_blur")
    metallic_strength = dpg.get_value("metallic_strength")
    if last_gray is None:
        return
    
    # Generate all maps
    normal_map = generate_normal_map(last_gray, norm_int)
    displacement_map = generate_displacement_visualization(last_gray, disp_scale)  # Use visualization version
    roughness_map = generate_roughness_map(last_gray, rough_contrast)
    ao_map = generate_ao_map(last_gray, ao_blur)
    metallic_map = generate_metallic_map(last_gray, metallic_strength)
    
    # Update all previews
    for tag, pilmap in [
        ("norm_tex", normal_map.convert("RGBA")),
        ("disp_tex", displacement_map.convert("RGBA")),
        ("rough_tex", roughness_map.convert("RGBA")),
        ("ao_tex", ao_map.convert("RGBA")),
        ("metallic_tex", metallic_map.convert("RGBA"))
    ]:
        arr = np.array(pilmap).astype(np.float32) / 255.0
        dpg.set_value(tag, arr.flatten())
    
    # Update PBR preview
    pbr_preview = generate_pbr_preview()
    dpg.set_value("pbr_preview_tex", pbr_preview.flatten())

def slider_callback(sender, app_data, user_data):
    update_all_maps_and_previews()

def export_all_maps():
    global normal_map, displacement_map, roughness_map, ao_map, metallic_map, export_directory
    if not export_directory:
        dpg.show_item("export_dir_dialog")
        return
    base_name = dpg.get_value("export_filename")
    if not base_name:
        base_name = "texture"
    
    # Generate the actual displacement map for export (not the visualization)
    disp_scale = dpg.get_value("disp_scale")
    actual_displacement = generate_displacement_map(last_gray, disp_scale)
    
    for map_type, img in {
        "normal": normal_map,
        "displacement": actual_displacement,  # Export actual displacement
        "roughness": roughness_map,
        "ao": ao_map,
        "metallic": metallic_map
    }.items():
        if img is not None:
            img.save(f"{export_directory}/{base_name}_{map_type}.png")
    dpg.show_item("export_success")

def export_dir_selected_callback(sender, app_data, user_data):
    global export_directory
    export_directory = app_data['file_path_name']
    dpg.set_value("export_dir_text", f"Export Dir: {export_directory}")
    save_config(input_directory, export_directory)

def update_cube_rotation():
    """Update cube rotation if cube mode is selected"""
    if dpg.get_value("preview_mode") == "Cube 3D":
        pbr_preview = generate_pbr_preview()
        dpg.set_value("pbr_preview_tex", pbr_preview.flatten())

def setup_cube_timer():
    """Setup a timer to update the rotating cube"""
    import threading
    import time
    
    def timer_thread():
        while dpg.is_dearpygui_running():
            if dpg.get_value("preview_mode") == "Cube 3D":
                try:
                    update_cube_rotation()
                except:
                    pass  # Ignore errors if UI is being destroyed
            time.sleep(1/30)  # 30 FPS
    
    timer = threading.Thread(target=timer_thread, daemon=True)
    timer.start()

# ---- DPG UI ----

dpg.create_context()

with dpg.texture_registry():
    dpg.add_dynamic_texture(img_size, img_size, np.zeros((img_size*img_size*4), dtype=np.float32), tag="norm_tex")
    dpg.add_dynamic_texture(img_size, img_size, np.zeros((img_size*img_size*4), dtype=np.float32), tag="disp_tex")
    dpg.add_dynamic_texture(img_size, img_size, np.zeros((img_size*img_size*4), dtype=np.float32), tag="rough_tex")
    dpg.add_dynamic_texture(img_size, img_size, np.zeros((img_size*img_size*4), dtype=np.float32), tag="ao_tex")
    dpg.add_dynamic_texture(img_size, img_size, np.zeros((img_size*img_size*4), dtype=np.float32), tag="metallic_tex")
    dpg.add_dynamic_texture(preview_size, preview_size, np.zeros((preview_size*preview_size*4), dtype=np.float32), tag="pbr_preview_tex")

with dpg.file_dialog(show=False, tag="file_dialog_id", default_path=input_directory, callback=load_texture_callback):
    dpg.add_file_extension("Images (*.png;*.jpg;*.jpeg){.png,.jpg,.jpeg}")
    dpg.add_file_extension("All Files (*.*){.*}")

with dpg.file_dialog(directory_selector=True, show=False, tag="export_dir_dialog", default_path=export_directory):
    dpg.add_file_extension("All Directories (*){.*}")
dpg.set_item_callback("export_dir_dialog", export_dir_selected_callback)

with dpg.window(label="Texture Map Generator", width=1900, height=1050):  # Taller window
    # Top section - file operations
    with dpg.group(horizontal=True):
        with dpg.child_window(width=480, height=120):
            dpg.add_text("File Operations")
            dpg.add_button(label="Load Image", callback=lambda: dpg.show_item("file_dialog_id"))
            dpg.add_text("No file loaded", tag="last_loaded_path", wrap=460)
            dpg.add_input_text(label="Export Filename", default_value="texture", tag="export_filename", width=300)
        
        with dpg.child_window(width=480, height=120):
            dpg.add_text("Export Settings")
            dpg.add_button(label="Select Export Directory", callback=lambda: dpg.show_item("export_dir_dialog"))
            dpg.add_text(f"Export Dir: {export_directory}", tag="export_dir_text", wrap=460)
            dpg.add_button(label="Export All Maps", callback=export_all_maps)
            dpg.add_text("Exported!", show=False, tag="export_success")
    
    dpg.add_separator()
    
    # Main content area - BIGGER LAYOUT
    with dpg.group(horizontal=True):
        # Left side - texture maps in 2x3 grid (bigger maps)
        with dpg.group():
            dpg.add_text("Generated Maps")
            with dpg.group(horizontal=True):
                # First row - bigger windows
                for label, tex_tag, slider_tag, slider_label, slider_def, slider_min, slider_max in [
                    ("Normal Map", "norm_tex", "norm_intensity", "Intensity", 2.0, 0.1, 8.0),
                    ("Displacement", "disp_tex", "disp_scale", "Scale", 1.0, 0.1, 5.0),
                ]:
                    with dpg.child_window(width=img_size+60, height=img_size+130):  # Bigger windows
                        dpg.add_text(label)
                        dpg.add_image(tex_tag)
                        dpg.add_slider_float(label=slider_label, default_value=slider_def, min_value=slider_min, max_value=slider_max, tag=slider_tag, callback=slider_callback)
            
            with dpg.group(horizontal=True):
                # Second row
                for label, tex_tag, slider_tag, slider_label, slider_def, slider_min, slider_max in [
                    ("Roughness", "rough_tex", "rough_contrast", "Contrast", 1.0, 0.1, 4.0),
                    ("AO", "ao_tex", "ao_blur", "Blur", 31.0, 3.0, 61.0),
                ]:
                    with dpg.child_window(width=img_size+60, height=img_size+130):
                        dpg.add_text(label)
                        dpg.add_image(tex_tag)
                        dpg.add_slider_float(label=slider_label, default_value=slider_def, min_value=slider_min, max_value=slider_max, tag=slider_tag, callback=slider_callback)
            
            with dpg.group(horizontal=True):
                # Third row
                for label, tex_tag, slider_tag, slider_label, slider_def, slider_min, slider_max in [
                    ("Metallic", "metallic_tex", "metallic_strength", "Strength", 1.0, 0.1, 4.0),
                ]:
                    with dpg.child_window(width=img_size+60, height=img_size+130):
                        dpg.add_text(label)
                        dpg.add_image(tex_tag)
                        dpg.add_slider_float(label=slider_label, default_value=slider_def, min_value=slider_min, max_value=slider_max, tag=slider_tag, callback=slider_callback)
        
        # Right side - MUCH BIGGER PBR preview
        with dpg.child_window(width=550, height=600):  # Much bigger preview area
            dpg.add_text("PBR Preview")
            dpg.add_combo(["Sphere", "Square", "Height Viz", "Cube 3D"], default_value="Square", tag="preview_mode", callback=slider_callback, width=200)
            dpg.add_text("Cube 3D mode rotates automatically - Use mouse wheel to zoom", wrap=530)
            dpg.add_spacer(height=10)
            dpg.add_image("pbr_preview_tex")
            
            # Add some preview controls
            dpg.add_spacer(height=10)
            dpg.add_text("Preview Controls:")
            dpg.add_slider_float(label="Rotation Speed", default_value=0.5, min_value=0.0, max_value=2.0, tag="rotation_speed", width=200)
            dpg.add_slider_float(label="Cube Scale", default_value=0.6, min_value=0.2, max_value=1.0, tag="cube_scale", width=200)

dpg.create_viewport(title='Texture Map Generator', width=1920, height=1080)
dpg.setup_dearpygui()

# Load config after setup
input_directory, export_directory = load_config()
if dpg.does_item_exist("export_dir_text"):
    dpg.set_value("export_dir_text", f"Export Dir: {export_directory}")

# Start the cube timer
setup_cube_timer()

dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()