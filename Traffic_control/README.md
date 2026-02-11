# INTEGRATION OF REINFORCE INTO TRAFIC SIMULATION WITH SUMO

Commands 

Observe initial intersection without control : 

	sumo-gui -c intersection/1tls_2x2.sumocfg

<img width="653" alt="Screenshot 2025-04-10 at 09 51 56" src="https://github.com/user-attachments/assets/d4d84d75-17b6-4316-81f1-7895718ce602" />

Locate into project location :

  	cd Traffic_control

Train and save your reinforce agent : 

	python train.py

Visualize cumulated rewards over episodes with tensorboard tools: 

	tensorboard --logdir runs/reinforce

Observe the trained agent in [SUMO GUI](https://sumo.dlr.de/docs/sumo-gui.html) : 

	python observe.py 



