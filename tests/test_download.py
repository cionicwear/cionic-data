'''
This test runs download.py on a set of test collections. The resulting downloaded
files are then reloaded and plotted as figures for manual verification of correctness.
Each test collection is chosen for a certain set of characteristics described below.
The figures are stored as .png in a directory called plots/ in the same folder as
the downloaded files.

For example, DPN 10 figures would be saved to recordings/cionic/DPN/10/plots/.

cionic DPN 3 (https://cionic.com/cionic/studies/DPN/collections/3)
    Contains labels
        0 standby (confirmed no strides)
        1 unstimulated_walk
    Left and right sides

cionic DPN 10 (https://cionic.com/cionic/studies/DPN/collections/10)
    Contains labels
        0 standby (confirmed no strides)
        1 stim_walk
        2 standby (confirmed no strides)
        3 unstimulated_walk
    Left side only

cionic KHE 6 (https://cionic.com/cionic/studies/khe/collections/6)
    Contains labels
        0 standby (confirmed no strides)
        1 stim_walk
        2 standby (confirmed no strides)
        3 unstimulated_walk
    Right side only
'''

import os
import re
import subprocess

import matplotlib.pyplot as plt
import pandas as pd

TEST_CASES = [
    {
        'args': ['cionic', 'DPN', '-n', '3', '-c', 'fquat'],
        'expected_dir': 'recordings/cionic/DPN/3',
    },
    {
        'args': ['cionic', 'DPN', '-n', '10', '-c', 'fquat'],
        'expected_dir': 'recordings/cionic/DPN/10',
    },
    {
        'args': ['cionic', 'khe', '-n', '6', '-c', 'fquat'],
        'expected_dir': 'recordings/cionic/khe/6',
    },
]


def get_splits_file(segment_num, splits_files):
    for splits_file in splits_files:
        if segment_num in splits_file:
            return splits_file
    return None


def remove_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_full_streams(files, expected_dir):
    if not os.path.exists(os.path.join(expected_dir, 'plots')):
        os.makedirs(os.path.join(expected_dir, 'plots'))
    splits_files = [f for f in os.listdir(expected_dir) if 'paired_splits' in f]
    splits_data = {
        re.findall(r'_(\d{3})_', f)[-1]: pd.read_csv(os.path.join(expected_dir, f))
        for f in splits_files
    }
    for file in files:
        # get the segment number and splits file
        segment_num = re.findall(r'_(\d{3})_', file)[-1]

        df = pd.read_csv(os.path.join(expected_dir, file))
        columns = [c for c in df.columns.tolist() if c != 'elapsed_s']
        _, axs = plt.subplots(
            len(columns), 1, figsize=(10, 2 * len(columns)), sharex=True
        )
        for i, column in enumerate(columns):
            axs[i].plot(df['elapsed_s'], df[column], label=column)
            axs[i].set_title('')
            axs[i].set_xlabel('Elapsed Time (s)')
            axs[i].set_ylabel('Value')
            axs[i].legend(loc='upper right')
            remove_spines(axs[i])

        splits = splits_data.get(segment_num)
        if splits is not None:
            for ax in axs:
                for _, pair in splits.iterrows():
                    ax.axvspan(pair['start'], pair['stop'], color='C0', alpha=0.1)

        plt.suptitle(file[:-4])  # Remove .csv extension from title
        plt.tight_layout()
        plt.savefig(os.path.join(expected_dir, 'plots', file[:-4] + '.png'))
        plt.close()


def plot_split_streams(files, expected_dir):
    if not os.path.exists(os.path.join(expected_dir, 'plots')):
        os.makedirs(os.path.join(expected_dir, 'plots'))
    for file in files:
        df = pd.read_csv(os.path.join(expected_dir, file))
        _, ax = plt.subplots(figsize=(10, 5))
        for i in range(len(df)):
            ax.plot(df.iloc[i, :], lw=0.8, color='C0', alpha=0.5)
        ax.set_title(file[:-4])  # Remove .csv extension from title
        ax.set_xticks(range(0, 101, 10), [f'{i}%' for i in range(0, 101, 10)])
        ax.set_xlim(0, 100)
        ax.set_xlabel('Percent of Gait Cycle')
        ax.set_ylabel('Value')
        remove_spines(ax)
        plt.tight_layout()
        plt.savefig(os.path.join(expected_dir, 'plots', file[:-4] + '.png'), dpi=150)
        plt.close()


def generate_test_plots(expected_dir):
    if not os.path.exists(expected_dir):
        raise FileNotFoundError(f'Expected directory {expected_dir} does not exist.')

    files = os.listdir(expected_dir)
    full_stream_files = [f for f in files if f.endswith('.csv') and 'splits' not in f]
    split_stream_files = [f for f in files if f.endswith('.csv') and 'splits' in f]
    print(f'Found {len(full_stream_files)} full stream CSV files in {expected_dir}')
    print(f'Found {len(split_stream_files)} splits CSV files in {expected_dir}')
    plot_full_streams(full_stream_files, expected_dir)
    plot_split_streams(split_stream_files, expected_dir)


def main():
    for test_case in TEST_CASES:
        subprocess.run(
            ['python3', 'scripts/download.py'] + test_case['args'], check=True
        )
        generate_test_plots(test_case['expected_dir'])


if __name__ == '__main__':
    main()
