{
    "mode": "sr",
    "use_cl": true,
    "gpu_ids": [0,1,2,3],

    "scale": 8,
    "is_train": true,
    // "use_chop": true,
    "use_chop" : false,
    "rgb_range": 255,
    "self_ensemble": false,
    "save_image": false,

    "datasets": {
        "train": {
            "mode": "LRHR",
            "data_path" : "Update the path respectively",
            //"data_path" : "/home/ashj/DEMO", 
            "dataroot_HR": "/mnt/data/paper99/DIV2K/Augment/DIV2K_train_HR_aug/x4",
            "dataroot_LR": "/mnt/data/paper99/DIV2K/Augment/DIV2K_train_LR_aug/x4",
            "data_type": "npy",
            "n_workers": 1,
            "batch_size": 4,
            "LR_size": 200,
            "use_flip": true,
            "use_rot": true,
            "noise": "."
        },
        "val": {
            "mode": "LRHR",
            "data_path" : "Update the path respectively",
            //"data_path" : "/home/ashj/DEMO", 
            "dataroot_HR": "./results/HR/Set5/x4",
            "dataroot_LR": "./results/LR/LRBI/Set5/x4",
            "data_type": "img",
            "n_workers": 1,
            "batch_size": 4,
            "LR_size": 200
        }
    },

    "networks": {
        "which_model": "SRFBN",
        "num_features": 64,
        "in_channels": 1,
        "out_channels": 1,
        "num_steps": 4,
        "num_groups": 8
    },

    "solver": {
        "type": "ADAM",
        "learning_rate": 0.0001,
        "weight_decay": 0,
        "lr_scheme": "MultiStepLR",
        "lr_steps": [45, 60, 70, 100],
        "lr_gamma": 0.5,
        "loss_type": "l1",
        "manual_seed": 0,
        "num_epochs": 1000,
        "skip_threshold": 3,
        "split_batch": 1,
        "save_ckp_step": 5,
        "save_vis_step": 1,
        "pretrain": null,
        "pretrained_path": "Update the path directing the model pretrained weights",
        "cl_weights": [1.0, 1.0, 1.0, 1.0]
    }
}
