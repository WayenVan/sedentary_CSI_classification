export PYTHONPATH := ./src:$$PYTHONPATH

all:

clean:
	rm -rf dist
	rm -rf src/*.egg-info

train_that_Timesdata: scripts/train_that_Timedata.py
	python3 scripts/train_that_Timedata.py \
	--epoch 60 \
	--batch 32 \
	--input_dim 1 \
	--save_dir models/THAT_timedata_timesdata \
	--input_dim 1 \
	--n_seq 3000 \
	--device cuda \
	--col_select timesData \
	--log 1

train_3channelGRU: scripts/training/train_channel3GRU_catm.py
	@echo start traning....
	python3 scripts/training/train_channel3GRU_catm.py
