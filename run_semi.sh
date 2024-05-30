CUDA_VISIBLE_DEVICES=3,4 \
python semi_train.py \
   --train_1 '/home/mdisk2/xukeke/CR_STR/datasets/train_dataset/label/CVPR2016'  \
   --train_2 '/home/mdisk2/xukeke/CR_STR/datasets/train_dataset/label/NIPS2014' \
   --unl_train_1 '/home/mdisk2/xukeke/CR_STR/datasets/data_CVPR2021/training/unlabel/U1.Book32' \
   --unl_train_2 '/home/mdisk2/xukeke/CR_STR/datasets/data_CVPR2021/training/unlabel/U2.TextVQA' \
   --unl_train_3 '/home/mdisk2/xukeke/CR_STR/datasets/data_CVPR2021/training/unlabel/U3.STVQA' \
   --valid_data '/home/mdisk2/xukeke/CR_STR/datasets/data_CVPR2021/evaluation/benchmark' \
   --eval_data '/home/mdisk2/xukeke/CR_STR/datasets/data_CVPR2021/evaluation' \
   --eval_type benchmark \
   --batchSize 768 \
   --unl_batchSize 768 \
   --model_name TRBA \
   --exp_name semi_exp_new_th_without_da\
   --Aug rand \
   --Aug_semi rand \
   --semi CrossEntropy \
   --workers 0 \
   --unl_workers 0 \
   --optimizer adamw \
   --weight_decay 0.01 \
   --lr 0.001 \
   --val_interval 300 \
   --data_usage_ratio 0.1 \
   --unlabel_data_usage_ratio 1 \
   --calibrator CAL \
   --alpha 0.1 \
   --exp_base 8 \
   --transit_time_ratio 0.2 \
   --num_iter 100000 \
   --checkpoint_root saved_models_fullchar \
   --language en \
   --robust True
   # --use_ada_threshold True \
   # --saved_model 'saved_models/TRBA/KLDiv/semi_exp_none_TSv1_5/300000_0.8__s.pth' \
   
   # 
   # --projection_type linear