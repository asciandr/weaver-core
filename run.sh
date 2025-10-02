#singularity shell -B /atlasgpfs01/usatlas/workarea/asciandra/ --nv colorsinglet.sif <<EOF
singularity shell -B /gpfs01/ --nv colorsinglet.sif <<EOF
export https_proxy=http://proxy.sdcc.bnl.local:3128/
export OMP_NUM_THREADS=4
export NCCL_DEBUG=INFO
export NCCL_P2P_LEVEL=NVL

cd weaver-core/

rm example_7labels.b*.yaml

wandb login

torchrun --standalone --nnodes=1 --nproc_per_node=4 -m weaver.train --data-train /gpfs01/usfcc/asciandra/training/output_submit_CheckNewGasMaterial_H*/subdir_*/out*.root --data-config example_7labels.yaml --network-config example_ParticleTransformer.py --model-prefix TRAINING_TEST_7labels_4GPUs --num-workers 1 --gpus 0,1,2,3 --batch-size 2048 --start-lr 1e-3 --num-epochs 1 --optimizer ranger --fetch-step 0.01 --backend nccl --log-wandb --wandb-displayname test_7labels_4GPU --wandb-projectname first_BNL_GPU_test_1GPU

EOF


#torchrun --standalone --nnodes=1 --nproc_per_node=4 -m weaver.train --data-train /atlasgpfs01/usatlas/workarea/asciandra/training/output_submit_wzp6_ee_nunuH_H*_ecm240_baseline/subdir_*/out_H*.root --data-config example.yaml --network-config example_ParticleTransformer.py --model-prefix TRAINING_TEST_4GPUs --num-workers 1 --gpus 1,2,3,4 --batch-size 2048 --start-lr 1e-3 --num-epochs 1 --optimizer ranger --fetch-step 0.01 --backend nccl --log-wandb --wandb-displayname test_4GPUs --wandb-projectname first_BNL_GPU_test_1GPU
#torchrun --standalone --nnodes=1 --nproc_per_node=4 -m weaver.train --data-train /atlasgpfs01/usatlas/workarea/asciandra/training/output_submit_wzp6_ee_nunuH_H*_ecm240_baseline/subdir_1*/out_H*.root /atlasgpfs01/usatlas/workarea/asciandra/training/output_submit_wzp6_ee_nunuH_H*_ecm240_baseline/subdir_2*/out_H*.root --data-config example.yaml --network-config example_ParticleTransformer.py --model-prefix TRAINING_TEST_4GPUs --num-workers 1 --gpus 1,2,3,4 --batch-size 2048 --start-lr 1e-3 --num-epochs 1 --optimizer ranger --fetch-step 0.01 --backend nccl --log-wandb --wandb-displayname test_4GPUs  --wandb-projectname first_BNL_GPU_test_1GPU
