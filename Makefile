all: clean build install

build: setup.py
	python -m build

install:
	pip3 install --force-reinstall dist/`ls dist | grep .whl`

uninstall:
	pip3 uninstall catm-package

clean:
	rm -rf dist
	rm -rf src/*.egg-info

train_that_default: scripts/train_that_Timedata.py
	python3 scripts/train_that_Timedata.py \
	--epoch 60 \
	--device cuda \
	--log 1


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