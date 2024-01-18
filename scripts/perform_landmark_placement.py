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
    parser.add_argument('--models_path', type=str, help="path to best models for each fold")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    fold_idx = np.load(args.splits_file, allow_pickle=True).item()
        
    # load pet and ct files
    pet_images = glob.glob(args.pet_files+'*.png')
    ct_images = glob.glob(args.ct_files+'*.png')
    
    # initialize relevant params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path_models = args.models_path
    
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
        

        # Load best model
        model_weights = torch.load(f"{path_models}/best_denoising_unet_fold{fold_id}.pt", map_location=device)
        model = nets.UNET(device,
                    batch_norm=True, 
                    sigma_noise=None,
                    input_channels=1,
                    nb_levels=4,
                    nb_filters=16,
                    output_channels=1).to(device)
        model.load_state_dict(model_weights)

        results = apply_landmarks(test_loader=test_loader, 
                                            model=model,
                                            face_mesh=face_mesh,
                                            mp_face_mesh=mp_face_mesh,
                                            mp_drawing=mp_drawing,
                                            mp_drawing_styles=mp_drawing_styles,
                                            device=device)
        
        # extract results
        results_ct = results["all_results_ct"]
        results_pet = results["all_results_pet"]
        results_pet_orig = results["all_results_pet_orig"]
        for it, (ct, pet, pet_orig) in enumerate(zip(results_ct, results_pet, results_pet_orig)):
            name_patient = pet_images[idx_test[it]].split("pets/")[1].split(".png")[0]
            # save landmarks positions
            try:
                face_ct = np.array([[res.x, res.y, res.z] for res in ct.multi_face_landmarks[0].landmark])
                np.save(f"../data/landmarks/fold_{fold_id}/{name_patient}_ct.npy", face_ct)
            except TypeError as err:
                print("Could not place landmarks in CT")

            try:
                face_pet = np.array([[res.x, res.y, res.z] for res in pet.multi_face_landmarks[0].landmark])
                np.save(f"../data/landmarks/fold_{fold_id}/{name_patient}_pet.npy", face_pet)

            except TypeError as err:
                print('Could not place landmarks in new PET')

            try:
                face_pet_orig = np.array([[res.x, res.y, res.z] for res in pet_orig.multi_face_landmarks[0].landmark])
                np.save(f"../data/landmarks/fold_{fold_id}/{name_patient}_pet_orig.npy", face_pet_orig)

            except TypeError as err:
                print("Could not place landmarks in original PET")