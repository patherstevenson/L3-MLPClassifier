set title 'MLP Accuracy'

set ylabel 'Accuracy'

set xlabel 'Nb categories'

set key autotitle columnheader

set term png size 800,600
set output 'images/hyperparameter_tuning.png'

plot for [col=2:4] 'dat/hyperparameter_tuning.dat' using 1:col with linespoints
