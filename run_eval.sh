
# # LongShortNet-S
# python tools/eval.py -f cfgs/longshortnet/s_s50_onex_dfp_tal_flip_s_1_d_1_l_3_d_1_yolox_shortcut_ep8 \
#                      -c /data/models/LongShortNet_release/s_s50_onex_dfp_tal_flip_s_1_d_1_l_3_d_1_yolox_shortcut_ep8/longshortnet_s.pth \
#                      -d 4 -b 16 --conf 0.01 --fp16


# # LongShortNet-M
# python tools/eval.py -f cfgs/longshortnet/m_s50_onex_dfp_tal_flip_s_1_d_1_l_3_d_1_yolox_shortcut_ep8 \
#                      -c /data/models/LongShortNet_release/m_s50_onex_dfp_tal_flip_s_1_d_1_l_3_d_1_yolox_shortcut_ep8/longshortnet_m.pth \
#                      -d 4 -b 16 --conf 0.01 --fp16


# # LongShortNet-L
# python tools/eval.py -f cfgs/longshortnet/l_s50_onex_dfp_tal_flip_s_1_d_1_l_3_d_1_yolox_shortcut_ep8 \
#                      -c /data/models/LongShortNet_release/l_s50_onex_dfp_tal_flip_s_1_d_1_l_3_d_1_yolox_shortcut_ep8/longshortnet_l.pth \
#                      -d 4 -b 16 --conf 0.01 --fp16


# LongShortNet-L 1200x1920
python tools/eval.py -f cfgs/longshortnet/l_s50_onex_dfp_tal_flip_s_1_d_1_l_3_d_1_yolox_shortcut_ep8_1200x1920 \
                     -c /data/models/LongShortNet_release/l_s50_onex_dfp_tal_flip_s_1_d_1_l_3_d_1_yolox_shortcut_ep8_1200x1920/longshortnet_l_1200x1920.pth \
                     -d 4 -b 16 --conf 0.01 --fp16