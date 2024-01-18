import glob
import argparse
import os
import nibabel as nib
from skimage import measure
from PIL import Image
from tqdm import tqdm

# local imports
from pet2face.reconstruction.utils import create_mesh, generate_morpho, find_skin

def find_all_nii_files(petct_path):
    list_ct_res_names = [] 
    list_suv_names = []
    for path, subdirs, files in os.walk(petct_path):
        for fname in files:
            if fname.startswith('CTres'):
                list_ct_res_names.append(os.path.join(path, fname))

            elif fname.startswith('SUV'):
                list_suv_names.append(os.path.join(path, fname))
    return list_ct_res_names, list_suv_names

def process_data(file_name, is_pet=False):
    f = nib.load(file_name)
    data = f.get_fdata()
    otsu, labels = find_skin(data, is_pet=is_pet)
    morpho = generate_morpho(labels)
    vox_dim = f.header.get_zooms()
    verts, faces, normals, vals = measure.marching_cubes(morpho[:,:, -morpho.shape[-1]//4:], .5, spacing=vox_dim)
    proj = create_mesh(verts, 
                        faces, 
                        vox_dim, 
                        height=morpho.shape[-1]//8)
    return proj

def check_already_processed(fname, flist):
    for f in flist:
        if fname in f:
            return True
    return False

def parse_arguments():
    parser = argparse.ArgumentParser(description="Performs cross validation")
    parser.add_argument('--petct_path', type=str, help="path to pet/ct nii files")
    parser.add_argument('--save_dir', type=str, help="directory where to save results")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    save_dir = args.save_dir
    
    # get all ct and suv nii files to process
    list_ct_res_names, list_suv_names = find_all_nii_files(args.petct_path)
    print(f"Found all files. Processing {len(list_ct_res_names)} files")
    # check for the ones already processed
    already_processed_cts = glob.glob(f"{save_dir}cts/*.png")
    already_processed_pets = glob.glob(f"{save_dir}pets/*.png")
    
    for ct_file, pet_file in tqdm(zip(list_ct_res_names, list_suv_names)):
        
        # name of the output
        path_save = '_'.join(ct_file.split("PETCT_")[1].split('/')[:2])

        if check_already_processed(path_save, already_processed_cts) == False:
            proj_ct = process_data(ct_file)
            res = Image.fromarray(proj_ct)
            res.save(f"{save_dir}cts/PETCT_{path_save}.png", dpi=(300,300))
        
        else:
            continue

        if check_already_processed(path_save, already_processed_pets) == False:
            proj_pet = process_data(pet_file, is_pet=True)
            Image.fromarray(proj_pet).save(f"{save_dir}pets/PETCT_{path_save}.png", dpi=(300,300))
        
        else:
            continue