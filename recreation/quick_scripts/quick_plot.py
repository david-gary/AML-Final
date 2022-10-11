from matplotlib.style import context
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

results_dict = {"SVM-Linear": 0.8879, "SVM-Quadratic": 0.5816,
                "SVM-RBF": 0.7498, "Naive Bayes": 0.6551, "ES1D": 0.9401}

results = pd.DataFrame.from_dict(
    results_dict, orient='index', columns=['Accuracy'])


results_dict = {"SVM-Linear": 0.8192, "SVM-Quadratic": 0.6502,
                "SVM-RBF": 0.7432, "Naive Bayes": 0.5321, "ES1D Recreation": 0.2493}

my_results = pd.DataFrame.from_dict(
    results_dict, orient='index', columns=['Accuracy'])


def plot_results(my_results_df, orig_results_df):

    # Use seaborn to make two subplots of bargraphs, stacked vertically
    # leave space between the two plots
    # Do not share the x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=(
        10, 10), gridspec_kw={'height_ratios': [1, 1]})

    sns.set(style="whitegrid")
    sns.set_context("poster")
    sns.set_palette("Set2")
    # add grid background
    ax1.grid(True, color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    ax2.grid(True, color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

    # First subplot is the original results
    sns.barplot(x=orig_results_df.index, y=orig_results_df['Accuracy'],
                ax=ax1, palette="Set2")
    ax1.set_title('ES1D Results')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    # x-axis tick labels are rotated 45 degrees
    for item in ax1.get_xticklabels():
        item.set_rotation(45)

    # Second subplot is the new results
    sns.barplot(x=my_results_df.index, y=my_results_df['Accuracy'],
                ax=ax2, palette="Set2")

    ax2.set_title('Recreation Results')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1)
    # x-axis tick labels are rotated 45 degrees
    for item in ax2.get_xticklabels():
        item.set_rotation(45)

    # Tight layout to make sure the subplots are not overlapping
    plt.tight_layout()

    # Save the full figure
    plt.savefig('results.png', dpi=300, bbox_inches='tight')

    # Show the figure
    plt.show()


def main():

    results_dict = {"SVM-Linear": 0.8879, "SVM-Quadratic": 0.5816,
                    "SVM-RBF": 0.7498, "Naive Bayes": 0.6551, "ES1D": 0.9401}

    results = pd.DataFrame.from_dict(
        results_dict, orient='index', columns=['Accuracy'])

    results_dict = {"SVM-Linear": 0.8192, "SVM-Quadratic": 0.6502,
                    "SVM-RBF": 0.7432, "Naive Bayes": 0.5321, "ES1D Recreation": 0.2493}

    my_results = pd.DataFrame.from_dict(
        results_dict, orient='index', columns=['Accuracy'])

    plot_results(my_results, results)


if __name__ == '__main__':
    main()
