"""Two small plotting helpers: the confusion matrix of a trained model,
and the training curve from Lightning's CSV logger.

Both are conveniences for looking at a single run. For the numbers that
go in a table, use ``scripts/score_folds.py`` and
``scripts/per_class_analysis.py``, which pool over folds and seeds.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchmetrics import ConfusionMatrix


def plot_confusion_matrix(model, 
                          dataloader, 
                          num_classes, 
                          id2labels_dict=None, 
                          normalize='true', 
                          seaborn=True, 
                          **kwargs):
    """
    normalize: Normalization mode for confusion matrix. Choose from:

    None or 'none': no normalization (default)
    'true': normalization over the targets (most commonly used)
    'pred': normalization over the predictions
    'all': normalization over the whole matrix

    kwargs are for seaborn.heatmap. Supported kwargs are annot, fmt, cmap, fontsize, figsize
    """
    # unpack some kwargs for seaborn.heatmap
    annot = kwargs.get('annot', True)
    fmt = kwargs.get('fmt', '.1%')
    cmap = kwargs.get('cmap', 'Blues')
    figsize = kwargs.get('figsize', (10, 8))
    fontsize = kwargs.get('fontsize', 12)



    if normalize == True:
        normalize = 'true'
    elif normalize == False:
        normalize = None

    cmat = ConfusionMatrix(task='multiclass', num_classes=num_classes, normalize=normalize)

    trainer = pl.Trainer(logger=False)

    for batch in trainer.predict(model, dataloaders=dataloader):
        # predict_step returns a 5-tuple (pred, proba, labels, names, attention);
        # the confusion matrix only needs pred + labels.
        predicted_labels, _proba, labels, _names = batch[:4]
        cmat(predicted_labels, labels)

    confmat = np.array(cmat.compute())

    fig, ax = plt.subplots(figsize=figsize)

    if seaborn:
        try:
            import seaborn as sns
        except ImportError:
            seaborn = False
            print("Seaborn is not available. Using matplotlib instead.")
        else:
            sns.heatmap(confmat, annot=annot, fmt=fmt, cmap=cmap, ax=ax)
    
    #if seaborn parameter is False OR seaborn is not available
    if not seaborn:
        cmat.plot(ax=ax)
        fig.colorbar(ax.imshow(confmat), ax=ax)

    if id2labels_dict is not None:
        ax.set_xticklabels([id2labels_dict[str(i)] for i in range(num_classes)], rotation=45, fontsize=fontsize)
        ax.set_yticklabels([id2labels_dict[str(i)] for i in range(num_classes)], rotation=45, fontsize=fontsize)

    ax.set_xlabel("Predicted", fontsize=fontsize)
    ax.set_ylabel("True label", fontsize=fontsize)

    return confmat, fig, ax


def plot_CSVLogger(csv_path):
    metrics = pd.read_csv(csv_path)

    aggreg_metrics = []
    agg_col = 'epoch'
    for i, dfg in metrics.groupby(agg_col):
        agg = dict(dfg.mean())
        agg[agg_col] = i
        aggreg_metrics.append(agg)

    df_metrics = pd.DataFrame(aggreg_metrics)

    ax1 = df_metrics[['train_loss', 'val_loss']].plot(
        grid=True, legend=True, xlabel='Epoch', ylabel='Loss'
    )
    ax2 = df_metrics[['val_acc']].plot(
        grid=True, legend=True, xlabel='Epoch', ylabel='Acc'
    )
    
    return ax1, ax2
