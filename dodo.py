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

def task_train_bvp():
    return cmd(
        envs +
        'python3 scripts/training/train_bvp.py '+
        '--debug 0 '+
        '--batch 16 '+
        '--lr 0.002 '+
        '--d_model 128 '+
        '--dropout 0.0 '

    )
def task_train_bvp_lr():
    for lr in [.8, .6, .4, .2]:
        yield cmd(
            envs +
            'python3 scripts/training/train_bvp.py '+
            '--epochs 20 '+
            '--debug 1 '+
            '--batch 64 '+
            '--lr {} '.format(lr)+
            '--dropout 0.0 '+
            '--model_save_dir models/bvp_lr{} '.format(lr),
            
            name='lr{}'.format(lr)
        )
