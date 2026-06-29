### Week 3 — 2026-06-29

**Attended this week's meeting:** Yes

**Progress this week**
- Resolved Gazebo gzclient crash: It turned out to be a startup race condition bug between gzserver and gzclient. Created ~/gz_tb3.sh to delay gzclient launch until gzserver by claude. Gazebo 3D visualization is now stable.
- Acquired Matterport3D dataset, configured ETPNav evaluation pipeline on R2R-CE, and fixed some code issues to run on limited hardware.
- Ran evaluation on 5 unseen scenes (pLe4wQe7qrG, zsNo4HB9uLZ, TbHJrupSAjP, QUCTc6BB5sX, 2azQ1b91cZZ), totaling 1089 episodes. Overall results: SR 60.07%, SPL 53.18%, nDTW 66.10%, SDTW 50.51%, consistent with paper-reported values (SR 57%, SPL 49%), confirming successful baseline reproduction.

**Challenges & blockers**
- Gzserver and gzclient starting simultaneously (race condition bug), not GPU/driver issues. Fixed by sequential launch script made by claude.
- Evaluation on T400 (2GB VRAM): Not enough VRAM to fit both models and renderer. Only to run all models on CPU, and use GPU only for rendering. Slower at ~9s/ep.

**Next steps**
- Analyze failure cases from evaluation logs to identify common failure modes.

**Hours spent (optional)** 
- IDK dude im drowning  someone SAVE ME.

**Links (optional):**
- ETPNav paper: https://arxiv.org/abs/2304.03047
- ETPNav code: https://github.com/MarSaKi/ETPNav
- I bought the M3D dataset from others.

| Scene | Episodes | SR (%) | SPL (%) | nDTW (%) | SDTW (%) | Dist (m) | Path (m) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| pLe4wQe7qrG | 18 | 66.67 | 48.80 | 46.01 | 31.63 | 3.14 | 9.74 |
| zsNo4HB9uLZ | 300 | 69.67 | 63.52 | 76.45 | 60.27 | 3.10 | 9.61 |
| TbHJrupSAjP | 264 | 60.98 | 54.15 | 70.85 | 52.56 | 3.81 | 10.30 |
| QUCTc6BB5sX | 255 | 54.90 | 48.30 | 60.20 | 45.80 | 5.53 | 13.09 |
| 2azQ1b91cZZ | 252 | 52.40 | 45.20 | 56.10 | 42.80 | 5.65 | 11.70 |
| **Overall** | **1089** | **60.07** | **53.18** | **66.10** | **50.51** | **4.43** | **11.08** |

<img width="1920" height="1080" alt="_cgi-bin_mmwebwx-bin_webwxgetmsgimg__ MsgID=4909538562688983283 skey=@crypt_9e33c4a3_fe2deaf65f745f1bd0716a6ebb7407df mmweb_appid=wx_webfilehelper" src="https://github.com/user-attachments/assets/75c103f4-5f20-4cca-ade3-4f61c6aaf733" />
<img width="1920" height="1080" alt="_cgi-bin_mmwebwx-bin_webwxgetmsgimg__ MsgID=6854542985277822008 skey=@crypt_9e33c4a3_fe2deaf65f745f1bd0716a6ebb7407df mmweb_appid=wx_webfilehelper" src="https://github.com/user-attachments/assets/4183fc2f-ff53-4744-b139-a71b77440782" />
