"""
plot_training.py
================
Reads TensorBoard event files and saves training curve plots as PNGs.
Shows: training loss, validation loss, metrics over epochs.
Demonstrates the val_loss fix — curves should show stable decreasing
val_loss rather than the >1.0 oscillating values from before.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record

LOGS_DIR    = Path('models/logs')
RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)


def read_tfevents(log_dir: Path) -> dict:
    """Read all scalar values from TensorBoard event files."""
    data = {}
    event_files = list(log_dir.rglob('events.out.tfevents.*'))
    print(f'  Found {len(event_files)} event files in {log_dir}')
    for ef in event_files:
        print(f'  Reading: {ef}')

    for ef in event_files:
        try:
            for event in tf.compat.v1.train.summary_iterator(str(ef)):
                for v in event.summary.value:
                    if hasattr(v, 'simple_value'):
                        tag = v.tag
                        if tag not in data:
                            data[tag] = {'steps': [], 'values': []}
                        data[tag]['steps'].append(event.step)
                        data[tag]['values'].append(v.simple_value)
        except Exception as e:
            print(f'  Error reading {ef}: {e}')
            continue

    return data


def plot_stage(stage_name: str, log_dir: Path, metrics: list):
    """Plot training curves for one stage."""
    if not log_dir.exists():
        print(f'WARNING: {log_dir} not found, skipping')
        return

    data = read_tfevents(log_dir)
    if not data:
        print(f'WARNING: No event data found in {log_dir}')
        return

    print(f'\n{stage_name} — available metrics: {list(data.keys())}')

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    fig.suptitle(
        f'{stage_name} Training Curves\n'
        f'Best epoch marked with ★ | LR decay visible as step changes in loss',
        fontsize=12
    )

    for ax, (train_key, val_key, title, ylabel) in zip(axes, metrics):
        train_data = data.get(train_key, {})
        val_data   = data.get(val_key, {})

        if train_data:
            ax.plot(train_data['steps'], train_data['values'],
                    label='Train', color='steelblue', linewidth=1.5)
        if val_data:
            ax.plot(val_data['steps'], val_data['values'],
                    label='Validation', color='coral', linewidth=1.5)

            # Mark best epoch
            best_idx = int(np.argmin(val_data['values'])
                          if 'loss' in val_key
                          else np.argmax(val_data['values']))
            best_step  = val_data['steps'][best_idx]
            best_value = val_data['values'][best_idx]
            ax.axvline(best_step, color='green', linestyle='--', alpha=0.5)
            ax.annotate(f'★ Best\nepoch {best_step}\n{best_value:.4f}',
                       xy=(best_step, best_value),
                       xytext=(best_step + max(val_data['steps']) * 0.05, best_value),
                       fontsize=8, color='green')

        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = RESULTS_DIR / f'training_curves_{stage_name.lower().replace(" ", "_")}.png'
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')


if __name__ == '__main__':
    # Stage 1: Restoration
    plot_stage(
        stage_name='Stage 1 Restoration',
        log_dir=LOGS_DIR / 'restoration',
        metrics=[
            ('epoch_loss',     'epoch_val_loss', 'Total Loss (L2 + Content)', 'Loss'),
            ('epoch_mse',      'epoch_val_mse',  'MSE',                       'MSE'),
            ('epoch_mae',      'epoch_val_mae',  'MAE',                       'MAE'),
        ]
    )

    # Stage 2: Segmentation
    plot_stage(
        stage_name='Stage 2 Segmentation',
        log_dir=LOGS_DIR / 'segmentation',
        metrics=[
            ('epoch_loss',  'epoch_val_loss',  'Combined Loss (Dice + BCE)', 'Loss'),
            ('epoch_dice',  'epoch_val_dice',  'Dice Coefficient',           'Dice'),
            ('epoch_iou',   'epoch_val_iou',   'IoU',                        'IoU'),
        ]
    )