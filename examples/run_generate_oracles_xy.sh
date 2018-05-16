#!/bin/bash


TRAIN_FOLDER_1="../../motion_planning_datasets/alternating_gaps/train/"
TRAIN_FOLDER_2="../../motion_planning_datasets/bugtrap_forest/train/"
TRAIN_FOLDER_3="../../motion_planning_datasets/forest/train/"
TRAIN_FOLDER_4="../../motion_planning_datasets/gaps_and_forest/train/"
TRAIN_FOLDER_5="../../motion_planning_datasets/mazes/train/"
TRAIN_FOLDER_6="../../motion_planning_datasets/multiple_bugtraps/train/"
TRAIN_FOLDER_7="../../motion_planning_datasets/shifting_gaps/train/"
TRAIN_FOLDER_8="../../motion_planning_datasets/single_bugtrap/train/"
VALID_FOLDER_1="../../motion_planning_datasets/alternating_gaps/validation/"
VALID_FOLDER_2="../../motion_planning_datasets/bugtrap_forest/validation/"
VALID_FOLDER_3="../../motion_planning_datasets/forest/validation/"
VALID_FOLDER_4="../../motion_planning_datasets/gaps_and_forest/validation/"
VALID_FOLDER_5="../../motion_planning_datasets/mazes/validation/"
VALID_FOLDER_6="../../motion_planning_datasets/multiple_bugtraps/validation/"
VALID_FOLDER_7="../../motion_planning_datasets/shifting_gaps/validation/"
VALID_FOLDER_8="../../motion_planning_datasets/single_bugtrap/validation/"
TEST_FOLDER_1="../../motion_planning_datasets/alternating_gaps/test/"
TEST_FOLDER_2="../../motion_planning_datasets/bugtrap_forest/test/"
TEST_FOLDER_3="../../motion_planning_datasets/forest/test/"
TEST_FOLDER_4="../../motion_planning_datasets/gaps_and_forest/test/"
TEST_FOLDER_5="../../motion_planning_datasets/mazes/test/"
TEST_FOLDER_6="../../motion_planning_datasets/multiple_bugtraps/test/"
TEST_FOLDER_7="../../motion_planning_datasets/shifting_gaps/test/"
TEST_FOLDER_8="../../motion_planning_datasets/single_bugtrap/test/"

FILE_TYPE="json"

#python generate_oracles_xy.py --database_folders ${TRAIN_FOLDER_1} ${TRAIN_FOLDER_2} ${TRAIN_FOLDER_3} ${TRAIN_FOLDER_4} ${TRAIN_FOLDER_5} ${TRAIN_FOLDER_6} ${TRAIN_FOLDER_7} ${TRAIN_FOLDER_8}  --num_envs 250 --file_start_num 0 --file_type ${FILE_TYPE}
#python generate_oracles_xy.py --database_folders ${VALID_FOLDER_1} ${VALID_FOLDER_2} ${VALID_FOLDER_3} ${VALID_FOLDER_4} ${VALID_FOLDER_5} ${VALID_FOLDER_6} ${VALID_FOLDER_7} ${VALID_FOLDER_8}  --num_envs 100 --file_start_num 800 --file_type ${FILE_TYPE}
python generate_oracles_xy.py --database_folders ${TRAIN_FOLDER_1} --num_envs 2 --file_start_num 0 --file_type ${FILE_TYPE}
python generate_oracles_xy.py --database_folders ${VALID_FOLDER_1} --num_envs 2 --file_start_num 800 --file_type ${FILE_TYPE}

