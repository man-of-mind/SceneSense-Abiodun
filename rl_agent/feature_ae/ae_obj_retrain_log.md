[2026-07-09 14:26:46] object-weighted AE retrain START (seg0.3 heat8 reg5 recon0.05)
[2026-07-09 14:26:46] AE_obj b128
model loaded (frozen); high_channels=960; bottleneck=128
data: train=10911 val=2110
distill weights: {'seg_w': 0.3, 'heat_w': 8.0, 'reg_w': 5.0, 'recon_w': 0.05}
b128 ep=0 train_loss=1.2488 val_loss=0.6704 (seg=0.0233 heat=0.03490 reg=0.0648 recon=1.2079) 125s
b128 ep=1 train_loss=0.9653 val_loss=3.4509 (seg=0.0660 heat=0.15975 reg=0.3272 recon=10.3435) 123s
b128 ep=2 train_loss=2.8043 val_loss=6.1492 (seg=0.1259 heat=0.18728 reg=0.4396 recon=48.3037) 124s
b128 ep=3 train_loss=2.7153 val_loss=1.6492 (seg=0.0327 heat=0.08223 reg=0.1608 recon=3.5486) 126s
b128 ep=4 train_loss=1.2500 val_loss=0.9159 (seg=0.0212 heat=0.03989 reg=0.0926 recon=2.5465) 122s
b128 ep=5 train_loss=0.7817 val_loss=0.7036 (seg=0.0162 heat=0.03067 reg=0.0724 recon=1.8315) 122s
b128 ep=6 train_loss=0.5988 val_loss=0.5668 (seg=0.0126 heat=0.02555 reg=0.0584 recon=1.3354) 123s
b128 ep=7 train_loss=1.0852 val_loss=1.0729 (seg=0.0187 heat=0.04657 reg=0.1041 recon=3.4787) 121s
b128 ep=8 train_loss=0.6259 val_loss=0.5744 (seg=0.0114 heat=0.02507 reg=0.0581 recon=1.6028) 120s
b128 ep=9 train_loss=0.4928 val_loss=0.5193 (seg=0.0097 heat=0.02484 reg=0.0512 recon=1.2287) 123s
b128 ep=10 train_loss=1.0924 val_loss=0.7748 (seg=0.0150 heat=0.03342 reg=0.0753 recon=2.5341) 123s
b128 ep=11 train_loss=0.5349 val_loss=0.5798 (seg=0.0149 heat=0.02359 reg=0.0549 recon=2.2375) 122s
b128 ep=12 train_loss=0.7857 val_loss=0.4722 (seg=0.0092 heat=0.01949 reg=0.0477 recon=1.4984) 123s
b128 ep=13 train_loss=0.8241 val_loss=0.5106 (seg=0.0090 heat=0.02165 reg=0.0527 recon=1.4286) 121s
b128 ep=14 train_loss=0.4905 val_loss=0.4857 (seg=0.0080 heat=0.02158 reg=0.0494 recon=1.2764) 120s
b128 ep=15 train_loss=0.4637 val_loss=0.6083 (seg=0.0091 heat=0.02968 reg=0.0615 recon=1.2167) 122s
b128 ep=16 train_loss=0.5244 val_loss=0.5152 (seg=0.0088 heat=0.02246 reg=0.0526 recon=1.4018) 121s
b128 ep=17 train_loss=0.4770 val_loss=0.6803 (seg=0.0113 heat=0.03064 reg=0.0687 recon=1.7602) 121s
b128 ep=18 train_loss=0.7788 val_loss=0.5403 (seg=0.0125 heat=0.02320 reg=0.0548 recon=1.5414) 121s
b128 ep=19 train_loss=0.5103 val_loss=0.4263 (seg=0.0069 heat=0.01856 reg=0.0440 recon=1.1092) 123s
DONE b128: best val_loss=0.4263 -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/rl_agent/feature_ae/checkpoints/ae_b128_obj.pt
[2026-07-09 15:07:36] AE_obj b64
model loaded (frozen); high_channels=960; bottleneck=64
data: train=10911 val=2110
distill weights: {'seg_w': 0.3, 'heat_w': 8.0, 'reg_w': 5.0, 'recon_w': 0.05}
b64 ep=0 train_loss=1.2369 val_loss=0.7773 (seg=0.0273 heat=0.03996 reg=0.0765 recon=1.3368) 123s
b64 ep=1 train_loss=0.6511 val_loss=1.3325 (seg=0.0339 heat=0.07587 reg=0.1271 recon=1.5983) 121s
b64 ep=2 train_loss=0.5452 val_loss=0.6299 (seg=0.0185 heat=0.03371 reg=0.0584 recon=1.2524) 121s
b64 ep=3 train_loss=0.7925 val_loss=1.0285 (seg=0.0352 heat=0.04558 reg=0.0961 recon=3.4608) 122s
b64 ep=4 train_loss=0.5594 val_loss=0.4924 (seg=0.0142 heat=0.02308 reg=0.0477 recon=1.2960) 121s
b64 ep=5 train_loss=0.5646 val_loss=0.4258 (seg=0.0118 heat=0.01873 reg=0.0421 recon=1.2405) 121s
b64 ep=6 train_loss=0.4298 val_loss=0.5580 (seg=0.0118 heat=0.02619 reg=0.0569 recon=1.2029) 123s
b64 ep=7 train_loss=0.4175 val_loss=0.4049 (seg=0.0095 heat=0.01880 reg=0.0395 recon=1.0791) 121s
b64 ep=8 train_loss=0.3771 val_loss=0.4049 (seg=0.0148 heat=0.01833 reg=0.0402 recon=1.0602) 120s
b64 ep=9 train_loss=0.5299 val_loss=0.4160 (seg=0.0094 heat=0.01875 reg=0.0421 recon=1.0498) 121s
b64 ep=10 train_loss=0.4385 val_loss=0.4176 (seg=0.0104 heat=0.01881 reg=0.0417 recon=1.1134) 119s
b64 ep=11 train_loss=0.3647 val_loss=0.6071 (seg=0.0142 heat=0.03101 reg=0.0604 recon=1.0566) 119s
b64 ep=12 train_loss=1.0772 val_loss=0.7187 (seg=0.0161 heat=0.03119 reg=0.0708 recon=2.2048) 120s
b64 ep=13 train_loss=0.4308 val_loss=0.4756 (seg=0.0113 heat=0.02131 reg=0.0482 recon=1.2141) 120s
b64 ep=14 train_loss=0.3767 val_loss=0.4641 (seg=0.0211 heat=0.02099 reg=0.0458 recon=1.2169) 123s
b64 ep=15 train_loss=0.5174 val_loss=0.4388 (seg=0.0095 heat=0.02032 reg=0.0430 recon=1.1653) 121s
b64 ep=16 train_loss=0.3622 val_loss=0.3209 (seg=0.0077 heat=0.01364 reg=0.0321 recon=0.9754) 120s
b64 ep=17 train_loss=0.3290 val_loss=0.3449 (seg=0.0080 heat=0.01617 reg=0.0333 recon=0.9366) 122s
b64 ep=18 train_loss=0.4926 val_loss=0.4612 (seg=0.0107 heat=0.02230 reg=0.0459 recon=1.0008) 121s
b64 ep=19 train_loss=0.4485 val_loss=4.9277 (seg=0.0473 heat=0.27961 reg=0.4646 recon=7.0763) 122s
DONE b64: best val_loss=0.3209 -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/rl_agent/feature_ae/checkpoints/ae_b64_obj.pt
[2026-07-09 15:48:03] AE_obj b32
model loaded (frozen); high_channels=960; bottleneck=32
data: train=10911 val=2110
distill weights: {'seg_w': 0.3, 'heat_w': 8.0, 'reg_w': 5.0, 'recon_w': 0.05}
b32 ep=0 train_loss=1.1862 val_loss=0.6605 (seg=0.0340 heat=0.03199 reg=0.0637 recon=1.5146) 125s
b32 ep=1 train_loss=0.6652 val_loss=0.5698 (seg=0.0267 heat=0.02710 reg=0.0545 recon=1.4457) 123s
b32 ep=2 train_loss=0.6003 val_loss=0.5741 (seg=0.0323 heat=0.02799 reg=0.0543 recon=1.3744) 123s
b32 ep=3 train_loss=0.4880 val_loss=0.8290 (seg=0.0224 heat=0.04958 reg=0.0716 recon=1.3480) 121s
b32 ep=4 train_loss=0.5059 val_loss=0.4082 (seg=0.0175 heat=0.01857 reg=0.0386 recon=1.2279) 121s
b32 ep=5 train_loss=0.4849 val_loss=0.3619 (seg=0.0146 heat=0.01526 reg=0.0348 recon=1.2294) 121s
b32 ep=6 train_loss=0.3893 val_loss=0.4486 (seg=0.0161 heat=0.02140 reg=0.0428 recon=1.1732) 122s
b32 ep=7 train_loss=0.5059 val_loss=0.5540 (seg=0.0157 heat=0.02675 reg=0.0539 recon=1.3152) 120s
b32 ep=8 train_loss=0.4226 val_loss=0.4590 (seg=0.0156 heat=0.02124 reg=0.0442 recon=1.2677) 122s
b32 ep=9 train_loss=0.3818 val_loss=0.3905 (seg=0.0184 heat=0.01743 reg=0.0376 recon=1.1483) 121s
b32 ep=10 train_loss=0.3739 val_loss=0.4774 (seg=0.0141 heat=0.02446 reg=0.0444 recon=1.1112) 122s
b32 ep=11 train_loss=0.4887 val_loss=0.3724 (seg=0.0136 heat=0.01634 reg=0.0355 recon=1.2006) 119s
b32 ep=12 train_loss=0.4175 val_loss=0.4983 (seg=0.0160 heat=0.02299 reg=0.0495 recon=1.2365) 121s
b32 ep=13 train_loss=0.3210 val_loss=0.5034 (seg=0.0171 heat=0.02505 reg=0.0482 recon=1.1352) 120s
b32 ep=14 train_loss=0.3788 val_loss=0.5999 (seg=0.0194 heat=0.03208 reg=0.0554 recon=1.2131) 121s
b32 ep=15 train_loss=0.3317 val_loss=0.3008 (seg=0.0106 heat=0.01233 reg=0.0298 recon=1.0009) 121s
b32 ep=16 train_loss=0.5682 val_loss=0.3807 (seg=0.0119 heat=0.01659 reg=0.0375 recon=1.1416) 120s
b32 ep=17 train_loss=0.3232 val_loss=0.4209 (seg=0.0145 heat=0.01920 reg=0.0418 recon=1.0745) 122s
b32 ep=18 train_loss=0.3725 val_loss=0.4532 (seg=0.0140 heat=0.02199 reg=0.0437 recon=1.0883) 121s
b32 ep=19 train_loss=0.3142 val_loss=0.3746 (seg=0.0119 heat=0.01723 reg=0.0367 recon=0.9940) 122s
DONE b32: best val_loss=0.3008 -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/rl_agent/feature_ae/checkpoints/ae_b32_obj.pt
AE_OBJ_RETRAIN_DONE
[2026-07-09 16:28:34] object-weighted AE retrain END
