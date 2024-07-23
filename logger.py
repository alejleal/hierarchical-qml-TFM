import wandb

import seaborn as sns
import numpy as np

def init_wandb(project, group, subgroup, name, config):
    run = wandb.init(project=project, group=group, job_type=subgroup, name=name, config=config)
    
    return run


def draw_heatmap(res, name, cmap = 'viridis'):
    image_path = f'./images/heatmap_{name}.png'

    # heatmap mask
    mask = np.triu(np.ones_like(res, dtype=bool))

    heatmap = sns.heatmap(res.iloc[1:,:-1], cmap=cmap, vmin=0, vmax=1, annot=True, mask=mask[1:,:-1])

    figure = heatmap.get_figure()
    figure.savefig(image_path, dpi=400)
    
    wandb.log({'heatmap': wandb.Image(image_path)})