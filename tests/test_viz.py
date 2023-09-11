import csv
import matplotlib.pyplot as plt
import os
from typing import Sequence


def slugfy(text):
    return '_'.join(text.lower().split())

def log_graph(Xs, Ys, label, **kwargs):
        path = "tests/tmp"
        try:
            filename = kwargs["fname"]
        except KeyError:
            filename = slugfy(label)

        if not isinstance(Xs, Sequence):
            Xs = [Xs] * len(Ys)
        print(len(Xs), len(Ys))
        captions = kwargs.get('captions', '')
        marker = kwargs.get('marker', ['', '', '', ''])
        style = kwargs.get('linestyle', ['-', '--', '-.', ':'])
        width = kwargs.get('linewidth', [2] * 4)
        fig, ax = plt.subplots()
        for i, (x, y) in enumerate(zip(Xs, Ys)):
            ax.plot(x, y, 
                    label=captions[i], 
                    linestyle=style[i % len(style)], 
                    linewidth=width[i % len(width)], 
                    marker=marker[i % len(marker)])
        ax.set_title(label)
        ax.set_xlabel(kwargs.get('xname', 'coords'))
        # ax.set_aspect('equal')
        ax.grid(True, which='both')
        # seaborn.despine(ax=ax, offset=0)
        ax.legend()
        fig.savefig(os.path.join(path, filename))
        plt.close()

def plot_losses(filepath):
    columns = {}
    with open(filepath) as f:
        reader = csv.DictReader(f)
        captions = list(reader.fieldnames)
        for row in reader:
            for key in captions:
                value = float(row[key])
                try:
                    columns[key].append(value)
                except KeyError:
                    columns[key] = [value]
    print(list(columns.values()))
    # x = list(range(len(columns[captions[0]])))
    # for loss in captions:
    #     log_graph([x], [columns[loss]], 
    #                     loss,
    #                     category='loss',
    #                     captions=[loss.split()[0]],
    #                     xname='epochs')
    ys = list(columns.values())
    xs = [list(range(len(columns[captions[0]])))] * len(ys)
    log_graph(xs, ys, 
                "All losses", 
                captions=captions,
                category='loss',
                fname='all_losses',
                xname='epochs')
    
if __name__ == '__main__':
    plot_losses('runs/logs/20230911-1515_cossineMG/1-1_w4F_hf216_MEp300_hl1_pr2/losses.csv')