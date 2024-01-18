import torch
import numpy as np
from copy import deepcopy
from piqa import MS_SSIM
import glob
import os
from typing import Dict, List, Any
import mediapipe as mp
from torch.utils.data import DataLoader
from torchvision import transforms
import pdb
# our package
from pet2face.net import data
from pet2face.reconstruction.landmarks import apply_landmarks
from pet2face.net.nets import UNET

class PET2CT:
    def __init__(self,
                 ct_images_path:List[str],
                 pet_images_path:List[str],
                 fold_split_file:str=None,
                 n_folds:int=5,
                 batch_size:int=32,
                 lr:float=1e-2,
                 epochs:int=30,
                 checkpoint_dir:str="../model_checkpoints",
                 ):
        # store list of pet/ct images
        self.ct_images = np.array(glob.glob(ct_images_path +'*.png'))
        self.pet_images = np.array(glob.glob(pet_images_path + '*.png'))
        
        # define relevant params for model
        self.n_epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.checkpoint_dir = checkpoint_dir
        assert os.path.exists(checkpoint_dir), "Checkpoint directory does not exist"

        # generate or load splits
        self.n_folds = n_folds
        if fold_split_file:
            print("Loading precomputed splits")
            self.fold_splits = np.load(fold_split_file, allow_pickle=True).item()
        else:
            print("Computing new splits")
            self.fold_splits = self.compute_splits()
            np.save(f"{checkpoint_dir}/fold_splits.npy", self.fold_splits, allow_pickle=True)

        # set relevant params for mediapipe
        self.set_mediapipe_params()
        
    def set_mediapipe_params(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True,
                                        max_num_faces=1,
                                        refine_landmarks=True,
                                        min_detection_confidence=0.5)
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1,
                                                 min_detection_confidence=0.5)
        
    def compute_splits(self)->Dict[int, List]:
        """
        Compute split repartition for cross val
        """
        fold_idx = {}

        percentage_train = 70 #test & val are equal and take the rest of the data
        for fold in range(self.n_folds):

            # Split dataset
            n_images = len(self.ct_images)
            idx_train = np.random.choice(np.arange(n_images),
                                        size=percentage_train*n_images//100,
                                        replace=False)

            idx_test_val = np.setdiff1d(np.arange(n_images), idx_train)

            idx_test = np.random.choice(idx_test_val,
                                        size=len(idx_test_val)//2,
                                        replace=False)

            idx_val = np.setdiff1d(idx_test_val, idx_test)
            
            fold_idx[fold] = {"train":idx_train, "val":idx_val, "test":idx_test}
        return fold_idx
  

    def fit(self,
            fold_id:int)->Dict[int, float]:
        
        train_loader = self.generate_loaders(fold_id, key='train')
        val_loader = self.generate_loaders(fold_id, key='val')

        # initialize model
        model = UNET(self.device,
                    batch_norm=True, 
                    sigma_noise=None,
                    input_channels=1,
                    nb_levels=4,
                    nb_filters=16,
                    output_channels=1).to(self.device)

        optimizer = torch.optim.Adam(params=list(model.parameters()), lr=self.lr)

    
        ssim_loss = MS_SSIM(n_channels=1).to(self.device)
        min_loss = 1e6
        training_loss = {}
        validation_loss = {}

        for epoch in range(self.n_epochs):
            print(f"\nRunning epoch {epoch}")

            ## training step
            model.train()
            train_loss = []
            for _, data in enumerate(train_loader, 0):
                pets, cts = data["pet"].to(self.device), data["ct"].to(self.device)
                outs = model(pets)
                optimizer.zero_grad()
                loss = 1 - ssim_loss(outs, cts)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())

            training_loss[epoch] = np.mean(train_loss)
            print(f"Training Loss = {training_loss[epoch]}")

            ## Validation step
            model.eval()
            val_loss = []
            with torch.no_grad():
                for _, data in enumerate(val_loader, 0):
                    pets, cts = data["pet"].to(self.device), data["ct"].to(self.device)
                    outs = model(pets)
                    loss = 1 - ssim_loss(outs, cts)
                    val_loss.append(loss.item())
                
            current_loss = np.mean(val_loss)
            validation_loss[epoch] = current_loss
            print(f"Validation Loss = {current_loss}")

            if current_loss < min_loss:
                print('Saving best model')
                torch.save(model.state_dict(), f'{self.checkpoint_dir}/best_denoising_unet_fold{fold_id}.pt')
                min_loss = current_loss
                self.models[fold_id] = deepcopy(model)
        return training_loss, validation_loss
    
    def generate_loaders(self, 
                         fold_id:int,
                         key:str='train'
                         ):
        # get idx for given split
        idx_key = self.fold_splits[fold_id][key]

        # get corresponding ct/pet
        ct_key = self.ct_images[idx_key]
        pet_key = self.pet_images[idx_key]

        # Create datasets
        data_key = data.ImageDataset(ct_key,
                                pet_key,
                                transform=transforms.Compose([data.Crop(), data.Normalize(), data.ToTensor()]))      
        # Create dataloaders
        key_loader = DataLoader(data_key,
                                batch_size=self.batch_size,
                                shuffle=True,
                                num_workers=8)
        return key_loader

                

    def predict(self,
                fold_id:int
                )->Dict[str,Any]:
        best_model = self.models[fold_id]

        # create test loader
        test_loader = self.generate_loaders(fold_id, key='test')
        
        # compute perfs
        test_perfs = apply_landmarks(test_loader=test_loader,
                                    model=best_model,
                                    face_mesh=self.face_mesh,
                                    mp_face_mesh=self.mp_face_mesh,
                                    mp_drawing=self.mp_drawing,
                                    mp_drawing_styles=self.mp_drawing_styles,
                                    device=self.device)
        return test_perfs

