POSTFIX=_lensnerf
COMMON_OPTIONS='--exp_postfix '$POSTFIX' --render_test --eval_lpips_vgg --eval_lpips_alex --eval_ssim --eval_dists --dump_images --aperture_sample_rate 7 ' #--perturb_infocusZ '${ppp}
#RENDER_OPTIONS='--exp_postfix '$POSTFIX' --aperture_sample_rate 7 --render_video'  #--render_test --render_video_aperture --render_video_pose '
RENDER_OPTIONS='--exp_postfix '$POSTFIX' --aperture_sample_rate 7 --render_video_pose_wo_apt '

DATA_NAME=$1
MODE=$2

if [ ${MODE} == 'train' ]
then
    #representative
    python run.py --config configs/lensnerf/${DATA_NAME}_trainF4_testF22.py ${COMMON_OPTIONS}
#    python run.py --config configs/lensnerf/${DATA_NAME}_trainF22_testF22.py ${COMMON_OPTIONS}
#    python run.py --config configs/lensnerf/${DATA_NAME}_trainF4_testF4.py ${COMMON_OPTIONS}
#    python run.py --config configs/lensnerf/${DATA_NAME}_trainF22_testF4.py ${COMMON_OPTIONS}

    # self
    #python run.py --config configs/lensnerf/${DATA_NAME}_trainF4_testF4.py ${COMMON_OPTIONS}
#    python run.py --config configs/lensnerf/${DATA_NAME}_trainF5dot6_testF5dot6.py ${COMMON_OPTIONS}
#    python run.py --config configs/lensnerf/${DATA_NAME}_trainF8_testF8.py ${COMMON_OPTIONS}
#    python run.py --config configs/lensnerf/${DATA_NAME}_trainF22_testF22.py ${COMMON_OPTIONS}

    #low to high
    #python run.py --config configs/lensnerf/${DATA_NAME}_trainF4_testF22.py ${COMMON_OPTIONS}
#    python run.py --config configs/lensnerf/${DATA_NAME}_trainF5dot6_testF22.py ${COMMON_OPTIONS}
#    python run.py --config configs/lensnerf/${DATA_NAME}_trainF8_testF22.py ${COMMON_OPTIONS}

    # high to low
    #python run.py --config configs/lensnerf/${DATA_NAME}_trainF22_testF4.py ${COMMON_OPTIONS}
#    python run.py --config configs/lensnerf/${DATA_NAME}_trainF22_testF8.py ${COMMON_OPTIONS}
#    python run.py --config configs/lensnerf/${DATA_NAME}_trainF22_testF5dot6.py ${COMMON_OPTIONS}

    # rendering
    #python run.py --config configs/lensnerf/${DATA_NAME}_trainF22_testF22.py ${RENDER_OPTIONS}

    # mix
#    python run.py --config configs/lensnerf/${DATA_NAME}_trainFmix_testF22.py ${COMMON_OPTIONS}
#    python run.py --config configs/lensnerf/${DATA_NAME}_trainFmix_testF4.py ${COMMON_OPTIONS}
elif [ ${MODE} == 'inter' ]
then
    # low to high intermediate
#    python run.py --config configs/lensnerf/${DATA_NAME}_trainF4_testF5dot6.py ${COMMON_OPTIONS}
#    python run.py --config configs/lensnerf/${DATA_NAME}_trainF4_testF8.py ${COMMON_OPTIONS}
#    python run.py --config configs/lensnerf/${DATA_NAME}_trainF5dot6_testF8.py ${COMMON_OPTIONS}
#    python run.py --config configs/lensnerf/${DATA_NAME}_trainF5dot6_testF4.py ${COMMON_OPTIONS}
#    python run.py --config configs/lensnerf/${DATA_NAME}_trainF8_testF4.py ${COMMON_OPTIONS}
#    python run.py --config configs/lensnerf/${DATA_NAME}_trainF8_testF5dot6.py ${COMMON_OPTIONS}
    python run.py --config configs/lensnerf/${DATA_NAME}_trainFmix_testF8.py ${COMMON_OPTIONS}
#    python run.py --config configs/lensnerf/${DATA_NAME}_trainFmix_testF5dot6.py ${COMMON_OPTIONS}
elif [ ${MODE} == 'render' ]
then
    #python run.py --config configs/lensnerf/${DATA_NAME}_trainF22_testF22.py ${RENDER_OPTIONS}
    python run.py --config configs/lensnerf/${DATA_NAME}_trainF4_testF22.py ${RENDER_OPTIONS}
else
    echo ${MODE}
    
fi

echo 'Done'
exit 0
