
from filtering import *
from data_loader import *
from es1d_model import *
import sys


# command line arguments for main represent num_subjects
def main():
    # get the number of subjects to train on
    num_subjects = int(sys.argv[1])

    # create the model
    model = RecreationES1D(num_subjects)

    # get number of epocks to train on
    num_epochs = int(sys.argv[2])

    # initial data load, train and eval data are both tuples of (data, labels)
    train_data, eval_data = load_train_test_data(num_subjects)

    # filter the data
    # train_data and eval_data are both tuples of (data, labels)
    filt_train_data, filt_eval_data = filter_data(train_data, eval_data)

    # finish preprocessing the data
    # train_data and eval_data are both tuples of (data, labels)
    train_data, eval_data = welch_data(filt_train_data, filt_eval_data)

    # train the model
    for epoch in range(num_epochs):
        print("Epoch: ", epoch)
        # train the model for one epoch
        model.single_training_epoch(train_data)
        # validate the model
        model.single_validation_epoch(eval_data)

    # save the model
    model.save_model()
