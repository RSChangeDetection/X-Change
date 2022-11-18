#! /bin/bash
cd ..
python test.py --save_result --save_iou_map --root='test_samples/' --checkpoint_path='checkpoints/saves/res18_best.pth'
