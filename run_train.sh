
# # LongShortNet-S
# python tools/train.py -f cfgs/longshortnet/s_s50_onex_dfp_tal_flip_s_1_d_1_l_3_d_1_yolox_shortcut_ep8 \
#                       -c /The/path/to/your/yolox_s.pth \
#                       --experiment-name s_s50_onex_dfp_tal_flip_s_1_d_1_l_3_d_1_yolox_shortcut_ep8 \
#                       -d 4 -b 16 --fp16

# # LongShortNet-M
# python tools/train.py -f cfgs/longshortnet/m_s50_onex_dfp_tal_flip_s_1_d_1_l_3_d_1_yolox_shortcut_ep8 \
#                       -c /The/path/to/your/yolox_m.pth \
#                       --experiment-name m_s50_onex_dfp_tal_flip_s_1_d_1_l_3_d_1_yolox_shortcut_ep8 \
#                       -d 4 -b 16 --fp16

# # LongShortNet-L
# python tools/train.py -f cfgs/longshortnet/l_s50_onex_dfp_tal_flip_s_1_d_1_l_3_d_1_yolox_shortcut_ep8 \
#                       -c /The/path/to/your/yolox_l.pth \
#                       --experiment-name l_s50_onex_dfp_tal_flip_s_1_d_1_l_3_d_1_yolox_shortcut_ep8 \
#                       -d 4 -b 16 --fp16

# LongShortNet-L
python tools/train.py -f cfgs/longshortnet/l_s50_onex_dfp_tal_flip_s_1_d_1_l_3_d_1_yolox_shortcut_ep8_1200x1920 \
                      -c /The/path/to/your/yolox_l.pth \
                      --experiment-name l_s50_onex_dfp_tal_flip_s_1_d_1_l_3_d_1_yolox_shortcut_ep8_1200x1920 \
                      -d 4 -b 16 --fp16