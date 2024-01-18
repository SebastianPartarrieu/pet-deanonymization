import argparse
import glob
import mediapipe as mp
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from skimage.restoration import denoise_wavelet

# local imports
from pet2face.net import nets
from pet2face.net.data import ImageDataset, Crop, ToTensor, Normalize
from pet2face.reconstruction.landmarks import apply_landmarks

def parse_arguments():
    parser = argparse.ArgumentParser(description="Performs landmark placement")
    parser.add_argument('--ct_files', type=str, help="path to ct images")
    parser.add_argument('--pet_files', type=str, help="path to pet images")
    parser.add_argument('--splits_file', type=str, default=None, help="path to precomputed splits")
    parser.add_argument('--save_perfs_dir', type=str, help="Path where saving perfs")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    fold_idx = np.load(args.splits_file, allow_pickle=True).item()
    
    # determine if running with deep or non deep denoising
   
    print("Running landmark placement with non deep denoising")
    landmarks_folder = "landmarks_non_deep"
    test_perfs_non_deep = {}
   
    # load pet and ct files
    pet_images = glob.glob(args.pet_files+'*.png')
    ct_images = glob.glob(args.ct_files+'*.png')
    
    # initialize relevant params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # intialize mediapipe params
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh
    mp_face_detection = mp.solutions.face_detection
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,
                                    max_num_faces=1,
                                    refine_landmarks=True,
                                    min_detection_confidence=0.5)
    face_detection = mp_face_detection.FaceDetection(model_selection=1,
                                                min_detection_confidence=0.5)
    
    # initializing non deep model
    model = lambda x: denoise_wavelet(x, 
                                    method='BayesShrink', mode='soft',
                                    rescale_sigma=True,)
         
    for fold_id in fold_idx:
        print(f"Processing fold {fold_id}")
        idx_test = fold_idx[fold_id]['test']
        ct_test = np.array(ct_images)[idx_test]
        pet_test = np.array(pet_images)[idx_test]

        data_test = ImageDataset(ct_test,
                                pet_test,
                                transform=transforms.Compose([Crop(), Normalize(), ToTensor()]))

        test_loader = DataLoader(data_test,
                                batch_size=32,
                                shuffle=False,
                                num_workers=2)
        
        

        results = apply_landmarks(test_loader=test_loader, 
                                  model=model,
                                  face_mesh=face_mesh,
                                  mp_face_mesh=mp_face_mesh,
                                  mp_drawing=mp_drawing,
                                  mp_drawing_styles=mp_drawing_styles,
                                  device=device,
                                  non_deep_denoising=True)
    
        # deleting mediapipe outputs as not pickable
        del results["all_results_pet"]
        del results["all_results_ct"]
        del results["all_results_pet_orig"]
        test_perfs_non_deep[fold_id] = results
       
    np.save(f"{args.save_perfs_dir}/test_perfs_non_deep_denoising.npy", test_perfs_non_deep, allow_pickle=True)
