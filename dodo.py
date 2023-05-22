from doit.action import CmdAction

DOIT_CONFIG = {
    'backend':'sqlite3'
}
envs = 'export PYTHONPATH=PYTHONPATH:/home/jingyan/Documents/sedentary_CSI_classification/src;'
shell = '/usr/bin/zsh'

def cmd(cmd: str):
    return {
        'actions': [
            CmdAction(cmd, buffering=1, executable=shell)
        ],
        'verbosity': 2
    }

def task_train_bvp():
    return cmd(
        envs +
        'python3 scripts/training/train_bvp.py '+
        '--debug 0 '+
        '--batch 1 '+
        '--lr 1e-10'
    )

