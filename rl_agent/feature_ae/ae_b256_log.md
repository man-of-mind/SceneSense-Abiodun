[2026-07-09 16:42:10] b256 object-weighted AE START (heat8 reg5 seg0.3)
model loaded (frozen); high_channels=960; bottleneck=256
data: train=10911 val=2110
distill weights: {'seg_w': 0.3, 'heat_w': 8.0, 'reg_w': 5.0, 'recon_w': 0.05}
b256 ep=0 train_loss=1.4579 val_loss=1.3718 (seg=0.0216 heat=0.08362 reg=0.1243 recon=1.4935) 121s
b256 ep=1 train_loss=8.3557 val_loss=3.2868 (seg=0.0750 heat=0.07331 reg=0.1775 recon=35.8044) 120s
b256 ep=2 train_loss=2.2574 val_loss=2.1522 (seg=0.0558 heat=0.07640 reg=0.2092 recon=9.5649) 122s
b256 ep=3 train_loss=3.2481 val_loss=1.6804 (seg=0.0425 heat=0.06170 reg=0.1267 recon=10.8082) 123s
b256 ep=4 train_loss=1.2662 val_loss=1.0884 (seg=0.0262 heat=0.04710 reg=0.0999 recon=4.0850) 123s
b256 ep=5 train_loss=1.0570 val_loss=1.0421 (seg=0.0223 heat=0.05079 reg=0.1002 recon=2.5654) 122s
b256 ep=6 train_loss=1.3990 val_loss=1.0151 (seg=0.0206 heat=0.04257 reg=0.0978 recon=3.5844) 121s
b256 ep=7 train_loss=1.6062 val_loss=1.2550 (seg=0.0270 heat=0.05121 reg=0.1129 recon=5.4562) 141s
b256 ep=8 train_loss=0.7649 val_loss=0.6310 (seg=0.0140 heat=0.02989 reg=0.0614 recon=1.6140) 225s
b256 ep=9 train_loss=1.0045 val_loss=0.9836 (seg=0.0151 heat=0.04575 reg=0.0998 recon=2.2828) 224s
b256 ep=10 train_loss=0.7945 val_loss=0.6239 (seg=0.0110 heat=0.02987 reg=0.0631 recon=1.3277) 220s
b256 ep=11 train_loss=0.9725 val_loss=0.7674 (seg=0.0113 heat=0.03752 reg=0.0757 recon=1.7124) 223s
b256 ep=12 train_loss=0.6756 val_loss=0.7558 (seg=0.0253 heat=0.03345 reg=0.0758 recon=2.0288) 221s
b256 ep=13 train_loss=1.1080 val_loss=0.8152 (seg=0.0143 heat=0.03768 reg=0.0778 recon=2.4113) 220s
b256 ep=14 train_loss=0.8652 val_loss=1.0353 (seg=0.0117 heat=0.04949 reg=0.1073 recon=1.9860) 225s
b256 ep=15 train_loss=0.6588 val_loss=0.6485 (seg=0.0112 heat=0.03105 reg=0.0667 recon=1.2668) 222s
b256 ep=16 train_loss=1.0795 val_loss=0.9124 (seg=0.0222 heat=0.03854 reg=0.1007 recon=1.8756) 213s
b256 ep=17 train_loss=1.3160 val_loss=1.6001 (seg=0.0297 heat=0.07215 reg=0.1673 recon=3.5544) 221s
DONE b256: best val_loss=0.6239 -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/rl_agent/feature_ae/checkpoints/ae_b256_obj.pt
AE_B256_DONE
[2026-07-09 17:35:44] b256 END
