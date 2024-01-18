import argparse
import numpy as np
from pet2face.net.trainers import PET2CT

def parse_arguments():
    parser = argparse.ArgumentParser(description="Performs cross validation")
    parser.add_argument('--n_epochs', type=int, default=10, help="# epochs to train")
    parser.add_argument('--n_folds', type=int, default=5, help="# folds in crossval")
    parser.add_argument('--batch_size', type=int, default=32, help="batch size")
    parser.add_argument('--lr', type=float, default=1e-2, help="Learning rate")
    parser.add_argument('--ct_files', type=str, help="path to ct images")
    parser.add_argument('--pet_files', type=str, help="path to pet files")
    parser.add_argument('--splits_file', type=str, default=None, help="path to precomputed splits")
    parser.add_argument('--save_perfs_dir', type=str, help="path to save performances")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    
    # get args values
    n_folds = args.n_folds
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    lr = args.lr
    ct_files = args.ct_files
    pet_files = args.pet_files
    split_files = args.splits_file
    save_perfs_dir = args.save_perfs_dir
    
    # initialize trainer
    trainer = PET2CT(ct_images_path=ct_files,
                     pet_images_path=pet_files,
                     fold_split_file=split_files,
                     n_folds=n_folds,
                     epochs=n_epochs,
                     batch_size=batch_size,
                     lr=lr, 
                     checkpoint_dir="../model_checkpoints")
    
    # initialize perf dict
    train_loss_all = {}
    val_loss_all = {}
    test_perfs = {}
    
    # run cross val
    print(f'Performing cross validation with {n_folds} folds')
    for fold_id in range(n_folds):
        # train/val
        print(f"Fold {fold_id+1} : Training model")
        train_loss, val_loss = trainer.fit(fold_id)
        train_loss_all[fold_id] = train_loss
        val_loss_all[fold_id] = val_loss
        
        # predict on test set
        print(f"Fold {fold_id+1}: Testing best model")
        test_perfs[fold_id] = trainer.predict(fold_id=fold_id)
    
    # dump everything
    print("Saving perfs")
    np.save(f"{save_perfs_dir}/train_perfs.npy", train_loss_all, allow_pickle=True)
    np.save(f"{save_perfs_dir}/val_perfs.npy", val_loss_all, allow_pickle=True)
    np.save(f"{save_perfs_dir}/test_perfs.npy", test_perfs, allow_pickle=True)
    
    
        
        
    
        