#! /bin/bash
cd ..
python train.py --epoch=500 --lr=2e-4 --root='data/LEVIR-CD/'
python train.py --epoch=500 --lr=1e-4 --enable_x_cross --resume --checkpoint_path='checkpoints/run/Best_CD.pth' --root='data/LEVIR-CD/'
