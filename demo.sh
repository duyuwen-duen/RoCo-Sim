ORIGIN_DATA_NAME='rcooper'
DATA_NAME='standard_rcooper_mini'
HOME_PATH='/home/ubuntu/duyuwen/ro-co-sim'
########################################## rcooper 136 ##########################################
sim_num=5 # generate image number
ROAD_SEQ='136-137-138-139'
INSERT_NUM=8 # place car number
BEGIN_RANGE=0 # min coverage range
END_RANGE=0.6 # max coverage range
INSERT_TYPE="rcooper" # dataset name
BASE_ROAD=0 # base road id
BASE_ROAD_NAME='136' # base road name
GENERATE_DATA_NAME=${INSERT_NUM}_${INSERT_TYPE}_${BEGIN_RANGE}_${END_RANGE}
SIM_DATA_NAME=standard_sim_${GENERATE_DATA_NAME}
data_type=train # train or val
max_dis=50 # the max distance if simulated cars to camera

############################# data convert ################################
# down load rcooper dataset and convert to standard format
# python data_utils/convert/rcooper2standard.py --data_path ${HOME_PATH}/data/${ORIGIN_DATA_NAME} --save_path ${HOME_PATH}/data/${DATA_NAME} --road_seq_list ${ROAD_SEQ} --sim_num ${sim_num}


# ############################# data process ################################
DEVICES=11
export CUDA_VISIBLE_DEVICES=${DEVICES}
# DepthSAM to get the depth map of foreground
python roco_sim/DepthSAM.py --road_seq ${ROAD_SEQ} --home_path ${HOME_PATH} --dataset ${DATA_NAME} --sim_num ${sim_num} --data_type ${data_type} 

# prepare data for blender
python  data_utils/data_prepare/data_parsing.py --home_path ${HOME_PATH} --dataset ${DATA_NAME} --road_seq intersection --road_seq ${ROAD_SEQ}  --sim_num ${sim_num} --data_type ${data_type}


# # # sample all the possible position for car
python data_utils/data_prepare/get_possible_position.py --home_path ${HOME_PATH} --dataset ${DATA_NAME} --road_seq ${ROAD_SEQ} --base_road ${BASE_ROAD} --max_dis ${max_dis} --data_type ${data_type}

# # # MOAS: choose the best position for car
python roco_sim/MOAS/MOAS.py --insert_car_num ${INSERT_NUM} --insert_range "${BEGIN_RANGE},${END_RANGE}" --insert_way ${INSERT_TYPE} --road_seq ${ROAD_SEQ} --home_path ${HOME_PATH} --dataset ${DATA_NAME} --view_weight 4 --occupancy_weight 1 --enlarge_scale 1 --sim_num ${sim_num} --data_type ${data_type} --color_type common --save_name ${GENERATE_DATA_NAME}

# # # # # ################################data generate################################


# # # use blender to generate data
python blender.py --setting_type ${SIM_DATA_NAME} --gpu_list ${DEVICES} --name ${ROAD_SEQ} --each_gpu_blender_num 1 --home_dir ${HOME_PATH} --dataset_type ${DATA_NAME} --data_type ${data_type} --home_dir ${HOME_PATH}


################################## visualization check #######################################
python data_utils/data_prepare/convert_label.py --home_path ${HOME_PATH} --dataset ${DATA_NAME}  --sim_data_name ${SIM_DATA_NAME} --road_seq ${ROAD_SEQ} --base_road ${BASE_ROAD} --data_type ${data_type}
python data_utils/vis/draw_bbox.py --data_path ${HOME_PATH}/result/${DATA_NAME}/${SIM_DATA_NAME} --road_seq ${ROAD_SEQ} --save_path ${HOME_PATH}/result/${DATA_NAME}/${SIM_DATA_NAME}/vis_label --data_type ${data_type}
