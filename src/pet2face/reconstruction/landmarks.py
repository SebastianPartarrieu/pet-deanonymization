import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import glob
import pdb

from skimage.color import rgb2gray
import cv2
from skimage import measure
import seaborn as sns


def add_face_detection(mp_drawing,
                       results, 
                       image):
    if results.detections:
        counter = 0
        annotated_image = image.copy()
        for detection in results.detections:
            mp_drawing.draw_detection(annotated_image, detection)
            counter += 1
        return annotated_image, counter
    else:
        return image, 0
    

def add_facial_landmarks(mp_drawing,
                         mp_face_mesh,
                         mp_drawing_styles,
                         results, image):
    """
    Add facial landmarks to img for plotting
    """
    if results.multi_face_landmarks:
        counter = 0
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
        counter += 1
        return image, counter
    else:
        return image, 0

def normalization_eyes(results, mp_face_mesh):
    """
    Get the interocular distance in the ct img with landmarks
    """
    mesh = np.array([(p.x, p.y) for p in results.multi_face_landmarks[0].landmark])

    # find landmark indices for eyes
    left_eye_idx = np.unique(list(mp_face_mesh.FACEMESH_LEFT_EYE))
    right_eye_idx = np.unique(list(mp_face_mesh.FACEMESH_RIGHT_EYE))

    # get midpoint for left eye
    left_eye = sorted(mesh[left_eye_idx], key=lambda x: x[0])
    left_eye_pt = (.5*(left_eye[-1][0] + left_eye[0][0]), .5*(left_eye[-1][1] + left_eye[0][1]))

    # get midpoint for right eye
    right_eye = sorted(mesh[right_eye_idx], key=lambda x: x[0])
    right_eye_pt = (.5*(right_eye[-1][0] + right_eye[0][0]), .5*(right_eye[-1][1] + right_eye[0][1]))

    # get normalized distance between both midpoints
    return right_eye_pt, left_eye_pt, np.linalg.norm(np.array(right_eye_pt) - np.array(left_eye_pt))

def get_distance_landmarks(res_ct, res_pet, mp_face_mesh, iod=None):
    """
    Implement the mean absolute distance betwwen the vertex locations,
    normalized by the distance between the eye centers (Interocular distance)
    """
    if res_ct.multi_face_landmarks is None or res_pet.multi_face_landmarks is None:
        print("Cannot compute the distance")
        return 

    # don't use normalization
    if iod is None:
        iod = 1

    mesh_ct = np.array([(p.x, p.y) for p in res_ct.multi_face_landmarks[0].landmark])
    mesh_pet = np.array([(p.x, p.y) for p in res_pet.multi_face_landmarks[0].landmark])
    return 100*np.sum(np.abs(mesh_ct - mesh_pet))/(iod*mp_face_mesh.FACEMESH_NUM_LANDMARKS)

def plot(img1, img2, title1, title2):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(img1, cmap='gray')
    axs[0].axis('off')
    axs[0].set(title=title1)

    axs[1].imshow(img2, cmap='gray')
    axs[1].axis('off')
    axs[1].set(title=title2)
    plt.show()

# Apply landmark detection
def apply_landmarks(test_loader, 
                    model,
                    face_mesh, 
                    mp_face_mesh,
                    mp_drawing,
                    mp_drawing_styles,
                    device,
                    non_deep_denoising=False):
    
    # store landmark placement
    all_results_ct = []
    all_results_pet = []
    all_results_pet_orig = []
    
    # store stats
    landmarks_detected_pet = 0
    landmarks_detected_pet_orig = 0
    landmarks_detected_ct = 0
    
    # store mean absolute distances
    mad_list = []
    mad_list_orig = []

    with torch.no_grad():
        for i, sample in enumerate(test_loader, 0):
            pet_orig_imgs = sample["pet"].numpy()
            if non_deep_denoising:
                pet_imgs = []
                for pet in pet_orig_imgs:
                    pet_imgs.append(model(pet.squeeze(0))[None, :, :])
                pet_imgs = np.vstack(pet_imgs)
                # to adapt to format expected by torch
                pet_imgs = torch.Tensor(pet_imgs[:, None, None, :, :]).to(device)
            else:
                pet_imgs = model(sample["pet"].to(device))
            ct_imgs = sample["ct"].numpy()
            for b in range(pet_imgs.shape[0]):
                ct_img = ct_imgs[b].squeeze(0)
                pet_orig_img = pet_orig_imgs[b].squeeze(0)
                pet_img = pet_imgs[b].squeeze(0).squeeze(0).cpu().numpy()

                # Transform to uint
                pet_orig_img = (pet_orig_img*255).astype(np.uint8)
                pet_img = (pet_img*255).astype(np.uint8)
                ct_img = (ct_img*255).astype(np.uint8)

                # Transform to RGB
                pet_img = cv2.cvtColor(pet_img, cv2.COLOR_GRAY2RGB)
                pet_orig_img = cv2.cvtColor(pet_orig_img, cv2.COLOR_GRAY2RGB)
                ct_img = cv2.cvtColor(ct_img, cv2.COLOR_GRAY2RGB)

                #facial landmarks
                results_ct = face_mesh.process(ct_img)
                results_pet = face_mesh.process(pet_img)
                results_pet_orig = face_mesh.process(pet_orig_img)
                
                # store results
                all_results_ct.append(results_ct)
                all_results_pet.append(results_pet)
                all_results_pet_orig.append((results_pet_orig))
                #add to plot
                ct_img, counter_landmarks_ct = add_facial_landmarks(mp_drawing,
                                                mp_face_mesh,
                                                mp_drawing_styles,
                                                results_ct, ct_img)

                pet_img, counter_landmarks_pet = add_facial_landmarks(mp_drawing,
                                                mp_face_mesh,
                                                mp_drawing_styles,
                                                results_pet, pet_img)

                pet_orig_img, counter_landmarks_pet_orig = add_facial_landmarks(mp_drawing,
                                                mp_face_mesh,
                                                mp_drawing_styles,
                                                results_pet_orig, pet_orig_img)


                #update
                if counter_landmarks_ct == 1:
                    landmarks_detected_ct += 1


                if counter_landmarks_pet == 1:
                    landmarks_detected_pet += 1

                if counter_landmarks_pet_orig == 1:
                    landmarks_detected_pet_orig += 1

                if (counter_landmarks_ct == 1) and (counter_landmarks_pet == 1):
                    right_eye_pt, left_eye_pt, iod = normalization_eyes(results_ct, mp_face_mesh)
                    mad = get_distance_landmarks(results_ct, results_pet, mp_face_mesh, iod)
                    mad_list.append(mad)
                    if (i % 100 == 0) and (b == 0):
                        print(f"Mean Absolute Distance between denoised PET & CT landmarks = {mad} %")

                if (counter_landmarks_ct == 1) and (counter_landmarks_pet_orig == 1):
                    right_eye_pt, left_eye_pt, iod = normalization_eyes(results_ct, mp_face_mesh)
                    mad = get_distance_landmarks(results_ct, results_pet_orig, mp_face_mesh, iod)
                    mad_list_orig.append(mad)
                    if (i % 100 == 0) and (b == 0):
                        print(f"Mean Absolute Distance between PET & CT landmarks = {mad} %")

    return {"landmarks_pet": landmarks_detected_pet,
        "landmarks_pet_orig": landmarks_detected_pet_orig,
        "landmarks_ct" :landmarks_detected_ct,
        "MAD":mad_list,
        "MAD_orig":mad_list_orig,
        "all_results_ct":all_results_ct,
        "all_results_pet":all_results_pet,
        "all_results_pet_orig":all_results_pet_orig
        }


def plot_landmarks(dl, 
                    size_test,
                    palette=sns.color_palette('husl')):

    # Compute proportion good landmarks
    good_landmarks_pet = np.sum(np.array(dl["MAD"]) < 15.)
    good_landmarks_pet /= size_test

    good_landmarks_pet_orig = np.sum(np.array(dl["MAD_orig"]) < 15.)
    good_landmarks_pet_orig /= size_test

    height = [dl['landmarks_ct']/size_test,
            dl['landmarks_pet_orig']/size_test,
            dl['landmarks_pet']/size_test,
            good_landmarks_pet_orig,
            good_landmarks_pet
            ]

    bars = ['CT landmarks',
            'PET landmarks',
            'PET denoised landmarks',
            'Good landmarks PET',
            'Good landmarks PET denoised'
        ]
    x_pos = [0., 1., 1.5, 2.5, 3.]

    plt.bar(x_pos, height,
            color=palette[:5],
            width=[.7, .5, .5, .5, .5])

    plt.xticks(x_pos, bars, rotation=45)
    plt.ylabel('%')
    plt.tight_layout()
    plt.show()