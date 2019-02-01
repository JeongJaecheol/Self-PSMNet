python2 train.py --maxdisp 192 \
		 --model stackhourglass_3d_share\
		 --datapath ../Depth-Estimation/data/Scene\ Flow\ Datasets/ \
		 --epoch 10 \
		 --batch_size 4 \
		 --savemodel ./trained \
		 --gpuids 0 1 2 3 \
		 --seed 1
