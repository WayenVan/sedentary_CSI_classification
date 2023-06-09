from doit.action import CmdAction

DOIT_CONFIG = {
    'backend':'sqlite3'
}
envs = 'export PYTHONPATH=PYTHONPATH:/home/jingyan/Documents/sedentary_CSI_classification/src;'
shell = '/usr/bin/zsh'

def cmd(cmd: str, base_name=None, name=None):
    ret =  {
        'actions': [
            CmdAction(cmd, buffering=1, executable=shell)
        ],
        'verbosity': 2
        }
    if base_name != None:
        ret['basename'] = base_name
    if name:
        ret['name'] = name
    return ret

def task_train_bvp_catm():
    return cmd(
        envs +
        'python3 scripts/training/train_bvp_catm.py '+
        '--debug 0 '+
        '--batch 32 '+
        '--lr 0.001 '+
        '--d_model 64 '+
        '--epochs 100 '+
        '--dropout 0.1 '
    )

def task_train_bvp_bvp():
    return cmd(
        envs +
        'python3 scripts/training/train_bvp_BVP.py '+
        '--device cuda '
        '--debug 0 '+
        '--batch 64 '+
        '--lr 0.01 '+
        '--d_model 64 '+
        '--epochs 20 '+
        '--dropout 0.1 '
    )

def task_train_3chanel_catm():
    return cmd(
        envs +
        'python3 scripts/training/train_channel3GRU_catm.py '+
        '--debug 0 '+
        '--batch 32 '+
        '--lr 0.005 '+
        '--d_model 64 '+
        '--epochs 100 '+
        '--dropout 0.1 '
    )