### Week 3 — 2026-06-28

**Attended this week's meeting:** Yes

**Progress this week**
- Resolved Gazebo gzclient crash: It turned out to be a startup race condition bug between gzserver and gzclient. Created ~/gz_tb3.sh to delay gzclient launch until gzserver by claude. Gazebo 3D visualization is now stable.
- Acquired Matterport3D dataset, configured ETPNav evaluation pipeline on R2R-CE, and fixed some code issues to run on limited hardware.
- Ran evaluation on 3 unseen scenes (zsNo4HB9uLZ, TbHJrupSAjP, pLe4wQe7qrG), totaling 582 episodes. Average results: SR 65.77%, SPL 55.49%, nDTW 64.44%, SDTW 48.15%, consistent with and slightly above paper-reported values (SR 57%, SPL 49%), confirming successful baseline reproduction.

**Challenges & blockers**
- Gzserver and gzclient starting simultaneously (race condition bug), not GPU/driver issues. Fixed by sequential launch script made by claude.
- Evaluation on T400 (2GB VRAM): Not enough VRAM to fit both models and renderer. Only to run all models on CPU, and use GPU only for rendering. Slower at ~9s/ep.

**Next steps**
- Run remaining QUCTc6BB5sX and 2azQ1b91cZZ to complete 5-scene subset evaluation.
- Analyze failure cases from evaluation logs to identify common failure modes.

**Hours spent (optional):** 
IDK dude im drowning  someone SAVE ME.

**Links (optional):**
- ETPNav paper: https://arxiv.org/abs/2304.03047
- ETPNav code: https://github.com/MarSaKi/ETPNav
- I bought the M3D dataset from others.

## Evaluation Results (ETPNav, val_unseen, 3 scenes)

| Scene | Episodes | SR (%) | SPL (%) | nDTW (%) | SDTW (%) | Dist (m) | Path (m) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| pLe4wQe7qrG | 18 | 66.67 | 48.80 | 46.01 | 31.63 | 3.14 | 9.74 |
| zsNo4HB9uLZ | 300 | 69.67 | 63.52 | 76.45 | 60.27 | 3.10 | 9.61 |
| TbHJrupSAjP | 264 | 60.98 | 54.15 | 70.85 | 52.56 | 3.81 | 10.30 |
| **Overall** | **582** | **65.77** | **55.49** | **64.44** | **48.15** | **3.35** | **9.88** |

<img width="1920" height="1080" alt="_cgi-bin_mmwebwx-bin_webwxgetmsgimg__ MsgID=4909538562688983283 skey=@crypt_9e33c4a3_fe2deaf65f745f1bd0716a6ebb7407df mmweb_appid=wx_webfilehelper" src="https://github.com/user-attachments/assets/75c103f4-5f20-4cca-ade3-4f61c6aaf733" />
<img width="1920" height="1080" alt="_cgi-bin_mmwebwx-bin_webwxgetmsgimg__ MsgID=6854542985277822008 skey=@crypt_9e33c4a3_fe2deaf65f745f1bd0716a6ebb7407df mmweb_appid=wx_webfilehelper" src="https://github.com/user-attachments/assets/4183fc2f-ff53-4744-b139-a71b77440782" />
