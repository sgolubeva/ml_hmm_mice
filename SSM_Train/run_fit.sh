#!/bin/bash
#Project Group: ml_mice
#Date Created: 2023-10-18

#SBATCH --account=bgmp                    #REQUIRED: which account to use
#SBATCH --partition=bgmp               #REQUIRED: which partition to use
#SBATCH --mail-user=sgolubev@uoregon.edu     #optional: if you'd like email
#SBATCH --mail-type=ALL                   #optional: must set email first, what type of email you want
#SBATCH --cpus-per-task=20                 #optional: number of cpus, default is 1
#SBATCH --mem=32GB                        #optional: amount of memory, default is 4GB

choices1=/projects/bgmp/shared/groups/2023/ml-mice/shared/mouse_behavior_for_GLMHMM/BW046_behavior_data/BW046_choices.npy
input1=/projects/bgmp/shared/groups/2023/ml-mice/shared/mouse_behavior_for_GLMHMM/BW046_behavior_data/BW046_inpts.npy
choices2=/projects/bgmp/shared/groups/2023/ml-mice/shared/mouse_behavior_for_GLMHMM/BW051_behavior_data/BW051_choices.npy
input2=/projects/bgmp/shared/groups/2023/ml-mice/shared/mouse_behavior_for_GLMHMM/BW051_behavior_data/BW051_inpts.npy
output1='BW046_n20_binary'
output2='BW051_n20_binary'

/usr/bin/time -v /projects/bgmp/shared/groups/2023/ml-mice/sgolubev/ml_hmm_mice/Mice_ML_prj/Test_data_and_scripts/mouse_behavior_for_GLMHMM/First_PT_Fitting_HMM/test_fitting_with_ray.py -i $input1 -c $choices1 -e $output1 -s 20
/usr/bin/time -v /projects/bgmp/shared/groups/2023/ml-mice/sgolubev/ml_hmm_mice/Mice_ML_prj/Test_data_and_scripts/mouse_behavior_for_GLMHMM/First_PT_Fitting_HMM/test_fitting_with_ray.py -i $input2 -c $choices2 -e $output2 -s 20