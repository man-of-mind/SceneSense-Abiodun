[2026-07-09 20:14:36] b32 clean-v2 train START
model loaded (frozen); high_channels=960; bottleneck=32
data: train=10911 val=2110
AE arch=v2  params=5,188,064
distill weights: {'seg_w': 0.3, 'heat_w': 8.0, 'reg_w': 5.0, 'recon_w': 0.05}
b32 ep=0 train_loss=0.5154 val_loss=0.1833 (seg=0.0135 heat=0.00646 reg=0.0160 recon=0.9510) 127s
b32 ep=1 train_loss=0.4795 val_loss=0.3511 (seg=0.0130 heat=0.01823 reg=0.0308 recon=0.9426) 124s
b32 ep=2 train_loss=0.1877 val_loss=0.1397 (seg=0.0083 heat=0.00492 reg=0.0111 recon=0.8434) 125s
b32 ep=3 train_loss=0.1513 val_loss=0.1164 (seg=0.0058 heat=0.00370 reg=0.0095 recon=0.7541) 122s
b32 ep=4 train_loss=0.1929 val_loss=0.1210 (seg=0.0058 heat=0.00433 reg=0.0097 recon=0.7235) 124s
b32 ep=5 train_loss=0.1135 val_loss=0.1144 (seg=0.0048 heat=0.00427 reg=0.0090 recon=0.6762) 124s
b32 ep=6 train_loss=0.1518 val_loss=0.0986 (seg=0.0045 heat=0.00324 reg=0.0077 recon=0.6557) 127s
b32 ep=7 train_loss=0.1042 val_loss=0.1125 (seg=0.0038 heat=0.00450 reg=0.0089 recon=0.6204) 122s
b32 ep=8 train_loss=0.1001 val_loss=0.0866 (seg=0.0036 heat=0.00274 reg=0.0067 recon=0.5978) 122s
b32 ep=9 train_loss=0.0877 val_loss=0.0976 (seg=0.0030 heat=0.00373 reg=0.0076 recon=0.5734) 122s
b32 ep=10 train_loss=0.0868 val_loss=0.0886 (seg=0.0032 heat=0.00320 reg=0.0068 recon=0.5572) 122s
b32 ep=11 train_loss=0.0808 val_loss=0.1052 (seg=0.0030 heat=0.00445 reg=0.0083 recon=0.5478) 121s
b32 ep=12 train_loss=0.0786 val_loss=0.0697 (seg=0.0029 heat=0.00207 reg=0.0052 recon=0.5247) 120s
b32 ep=13 train_loss=0.0736 val_loss=0.0670 (seg=0.0024 heat=0.00197 reg=0.0050 recon=0.5085) 124s
b32 ep=14 train_loss=0.0827 val_loss=0.0712 (seg=0.0026 heat=0.00231 reg=0.0053 recon=0.5068) 123s
b32 ep=15 train_loss=0.0696 val_loss=0.0657 (seg=0.0023 heat=0.00200 reg=0.0049 recon=0.4893) 123s
b32 ep=16 train_loss=0.0721 val_loss=0.0651 (seg=0.0021 heat=0.00196 reg=0.0049 recon=0.4849) 122s
b32 ep=17 train_loss=0.0641 val_loss=0.0737 (seg=0.0026 heat=0.00256 reg=0.0058 recon=0.4726) 122s
b32 ep=18 train_loss=0.0684 val_loss=0.0671 (seg=0.0026 heat=0.00220 reg=0.0050 recon=0.4712) 122s
b32 ep=19 train_loss=0.0616 val_loss=0.0842 (seg=0.0023 heat=0.00347 reg=0.0065 recon=0.4629) 122s
b32 ep=20 train_loss=0.0598 val_loss=0.0575 (seg=0.0023 heat=0.00168 reg=0.0042 recon=0.4436) 124s
b32 ep=21 train_loss=0.0604 val_loss=0.0593 (seg=0.0022 heat=0.00180 reg=0.0044 recon=0.4410) 122s
b32 ep=22 train_loss=0.0606 val_loss=0.0613 (seg=0.0020 heat=0.00203 reg=0.0045 recon=0.4345) 124s
b32 ep=23 train_loss=0.0560 val_loss=0.0568 (seg=0.0019 heat=0.00178 reg=0.0042 recon=0.4232) 124s
b32 ep=24 train_loss=0.0606 val_loss=0.0538 (seg=0.0020 heat=0.00157 reg=0.0039 recon=0.4209) 122s
b32 ep=25 train_loss=0.0543 val_loss=0.0567 (seg=0.0019 heat=0.00188 reg=0.0041 recon=0.4131) 124s
b32 ep=26 train_loss=0.0550 val_loss=0.0556 (seg=0.0020 heat=0.00173 reg=0.0041 recon=0.4120) 122s
b32 ep=27 train_loss=0.0541 val_loss=0.0597 (seg=0.0021 heat=0.00207 reg=0.0044 recon=0.4070) 125s
b32 ep=28 train_loss=0.0516 val_loss=0.0534 (seg=0.0021 heat=0.00164 reg=0.0039 recon=0.4004) 123s
b32 ep=29 train_loss=0.0517 val_loss=0.0564 (seg=0.0019 heat=0.00194 reg=0.0041 recon=0.3942) 124s
b32 ep=30 train_loss=0.0528 val_loss=0.0495 (seg=0.0019 heat=0.00142 reg=0.0036 recon=0.3908) 123s
b32 ep=31 train_loss=0.0500 val_loss=0.0528 (seg=0.0020 heat=0.00168 reg=0.0039 recon=0.3886) 122s
b32 ep=32 train_loss=0.0496 val_loss=0.0467 (seg=0.0017 heat=0.00129 reg=0.0034 recon=0.3821) 125s
b32 ep=33 train_loss=0.0500 val_loss=0.0852 (seg=0.0021 heat=0.00383 reg=0.0069 recon=0.3900) 123s
b32 ep=34 train_loss=0.0506 val_loss=0.0510 (seg=0.0017 heat=0.00167 reg=0.0037 recon=0.3758) 122s
b32 ep=35 train_loss=0.0472 val_loss=0.0530 (seg=0.0017 heat=0.00174 reg=0.0040 recon=0.3735) 123s
b32 ep=36 train_loss=0.0491 val_loss=0.0460 (seg=0.0018 heat=0.00128 reg=0.0033 recon=0.3701) 122s
b32 ep=37 train_loss=0.0470 val_loss=0.0473 (seg=0.0019 heat=0.00137 reg=0.0035 recon=0.3673) 122s
b32 ep=38 train_loss=0.0483 val_loss=0.0460 (seg=0.0020 heat=0.00129 reg=0.0034 recon=0.3654) 125s
b32 ep=39 train_loss=0.0461 val_loss=0.0663 (seg=0.0019 heat=0.00278 reg=0.0050 recon=0.3680) 124s
DONE b32: best val_loss=0.0460 -> /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/rl_agent/feature_ae/checkpoints/ae_b32_v2clean.pt
[2026-07-09T21:36:47] Evaluating on device=cuda (NVIDIA GeForce RTX 5090)
[2026-07-09T21:39:32] Evaluation split=test miou=0.8371 vehicle_iou=0.9317 learned_xy_mae=2.1108762284021916; metrics=/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/mprime_dropaware_20260708/sweeps/ae_v2clean_b32/metrics
wrote rl_agent/COMPLETE_KNOB_MATRIX.md  (18 profiles, 9 within tol)
AE_B32CLEAN_DONE
[2026-07-09 21:39:33] b32 clean-v2 END
