import numpy as np
from skimage.filters import threshold_otsu
import cc3d
import open3d as o3d
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
from skimage import measure
import nibabel as nib
import cv2

def find_skin(data, is_pet=False):
    if is_pet:
        otsu = np.percentile(data, 95)
        data_bin = (data > otsu)
    else:
        otsu = threshold_otsu(data)
        data_bin = (data >= otsu)
        
    labels = cc3d.connected_components(data_bin)
    return otsu, labels

def generate_morpho(lab):
    morpho = lab.copy()
    unique_lab, counts = np.unique(lab, return_counts=True)
    skin = unique_lab[np.argsort(counts)[-2]] # first largest comp is background
    morpho = (morpho == skin).astype(float)
    return morpho

def create_mesh(verts, faces, vox_dim, height, n_it=5):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    mesh = mesh.filter_smooth_laplacian(n_it)
    mesh.compute_vertex_normals()
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=90,
        center=[200*vox_dim[0], 0, height*vox_dim[-1]],
        eye=[200*vox_dim[0], 400*vox_dim[1], height*1.01*vox_dim[-1]],
        up=[0, 1, 0],
        width_px=512,
        height_px=512,
    )
    ans = scene.cast_rays(rays)
    proj = np.abs(ans['primitive_normals'].numpy())
    proj_g = (rgb2gray(proj)*255).astype(np.uint8)
    return proj_g

def create_mask(proj, enlarge=10):
    """
    Returns bounding box of head when detected
    """
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face = face_cascade.detectMultiScale(proj, scaleFactor=1.2, minNeighbors=5)
    if len(face) > 0:
        x, y, w, h = face[0]
        x = x - enlarge
        y = y - enlarge
        w = w + 2*enlarge
        h = h + 2*enlarge
        return x, y, w, h
    else:
        return 0


def calculate_loss(ct, pet, mask):
    x, y, w, h = mask
    ct_extract = ct[y:y+h, x:x+w]
    pet_extract = pet[y:y+h, x:x+w]
    rmse = np.linalg.norm(ct_extract - pet_extract)/np.sqrt(ct_extract.shape[0]*ct_extract.shape[1])
    sim = ssim(ct_extract, pet_extract)
    return rmse, sim

def process_data(file_name, is_pet=False):
    """
    Performs all steps required to build 2D morphology
    """
    f = nib.load(file_name)
    data = f.get_fdata()
    otsu, bin, labels = find_skin(data, is_pet=is_pet)
    morpho = generate_morpho(labels)
    vox_dim = f.header.get_zooms()
    verts, faces, normals, vals = measure.marching_cubes(morpho[:,:, -morpho.shape[-1]//4:], .5, spacing=vox_dim)
    proj = create_mesh(verts, 
                        faces, 
                        vox_dim, 
                        height=morpho.shape[-1]//8)
    return proj