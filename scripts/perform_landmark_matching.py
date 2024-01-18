import numpy as np
import argparse
import glob
import open3d as od
def parse_arguments():
    parser = argparse.ArgumentParser(description="Performs cross validation")
    parser.add_argument('--landmarks_files', type=str, help="path to ct landmarks files")
    parser.add_argument('--n_folds', type=str, help="# folds")
    parser.add_argument('--save_dir', type=str, help="directory where to save results")
    parser.add_argument('--icp', help="whether to add alignement step", action="store_true")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    n_folds = int(args.n_folds)
    save_dir = args.save_dir
    for fold in range(n_folds):
        print(f"Processing fold {fold}")
        pet_landmarks_path = glob.glob(f"{args.landmarks_files}/fold_{fold}/*pet.npy")
        ct_landmarks_path = glob.glob(f"{args.landmarks_files}/fold_{fold}/*ct.npy")
        pet_landmarks = [np.load(f) for f in pet_landmarks_path]
        ct_landmarks = [np.load(f) for f in ct_landmarks_path]
        
        if args.icp:
            print("Adding ICP alignment")
            all_distances = []
            for pet in pet_landmarks:
                distances_to_pet = []
                
                for ct in ct_landmarks:
                    # convert pet to point cloud
                    pet_pcd = od.geometry.PointCloud()
                    pet_pcd.points = od.utility.Vector3dVector(pet)

                    # store ct as point cloud
                    ct_pcd = od.geometry.PointCloud()
                    ct_pcd.points = od.utility.Vector3dVector(ct)

                    # align ct_sample
                    res_align = od.pipelines.registration.registration_icp(source=ct_pcd,
                                                                    target=pet_pcd,
                                                                    max_correspondence_distance=1.,
                                                                    )
                    # apply transformation
                    transfo = res_align.transformation
                    ct_pcd.transform(transfo)
                    ct_transformed = np.array(ct_pcd.points)
                    
                    # store distances
                    distances_to_pet.append(np.linalg.norm(ct_transformed - pet))
                all_distances.append(distances_to_pet)
            np.save(f"{save_dir}/distances_fold{fold}_aligned.npy", np.array(all_distances))
        else:
            print("Not performing ICP alignement")
            all_distances = []
            for pet in pet_landmarks:
                distances_to_pet = []
                for ct in ct_landmarks:
                    distances_to_pet.append(np.linalg.norm(pet - ct))
                all_distances.append(distances_to_pet)
            np.save(f"{save_dir}/distances_fold{fold}.npy", np.array(all_distances))