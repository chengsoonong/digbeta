#!/bin/bash
mv cliques_all_1.pkl.gz cliques_all.pkl.gz
rm cliques_all_[2-4].pkl.gz

mv cliques_trndev_1.pkl.gz cliques_trndev.pkl.gz
rm cliques_trndev_[2-4].pkl.gz

mv Y_1.pkl.gz Y.pkl.gz
rm Y_[2-4].pkl.gz

mv Y_train_1.pkl.gz Y_train.pkl.gz
rm Y_train_[2-4].pkl.gz

mv Y_trndev_1.pkl.gz Y_trndev.pkl.gz
rm Y_trndev_[2-4].pkl.gz
