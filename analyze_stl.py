import trimesh
import numpy as np

stl_path = r"S:\ComfyUI\output\preview_vtk_9f26de9e.stl"

try:
    mesh = trimesh.load(stl_path)
    print(f"Mesh loaded from {stl_path}")
    print(f"Vertices: {len(mesh.vertices)}")
    print(f"Faces: {len(mesh.faces)}")
    print(f"Bounding Box: {mesh.bounds}")

    # Calculate vertex distances from origin
    dist = np.linalg.norm(mesh.vertices, axis=1)
    print(
        f"Distance from origin: Min={dist.min():.4f}, Max={dist.max():.4f}, Mean={dist.mean():.4f}"
    )

    # Check for NaNs or Infs in vertices
    if not np.isfinite(mesh.vertices).all():
        print("CRITICAL: Mesh contains non-finite vertices!")

    # Area check
    print(f"Surface Area: {mesh.area}")

    if mesh.is_watertight:
        print("Mesh is watertight.")
    else:
        print("Mesh is not watertight (expected for ERP mesh with poles/seams).")

except Exception as e:
    print(f"Error analyzing mesh: {e}")
