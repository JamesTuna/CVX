
MU="0.001 0.01 0.1 1 10 100 1000 10000 100000"
for mu in $MU
do
  python3 logisticRegression.py --batch-size 32 --iter 60000 --MU $mu --optimizer GD --interval 3000 --lr 50
done
