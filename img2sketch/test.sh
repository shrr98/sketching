dataDir="$(pwd)/img2sketch"

python main.py \
    --name pretrained \
    --dataset_mode test_dir \
    --dataroot $1 \
    --file_name  $2 \
    --model pix2pix \
    --which_direction AtoB \
    --norm batch \
    --input_nc 3 \
    --output_nc 1 \
    --which_model_netG resnet_9blocks \
    --no_dropout
