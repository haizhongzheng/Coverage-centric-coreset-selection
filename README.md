
# Coverage-centric Coreset Selection for High Pruning Rates

Training cmd example
```
#Train the model with the entire dataset
python train.py --dataset cifar10 --gpuid 0 --epochs 200 --lr 0.1 --network resnet18 --batch-size 256 --task-name all-data --base-dir ./model/cifar10

#Calculate importance score for data
python generate_importance_score.py --gpuid 0 --base-dir ./model/cifar10 --task-name all-data

#Train model on a coreset with a 90% pruning rate
python train.py --dataset cifar10 --gpuid 0 --iterations 40000 --task-name forgetting-0.1 --base-dir ./model/cifar10/stratified --coreset --coreset-mode stratified --data-score-path ./model/cifar10/all-data/data-score-all-data.pickle --coreset-key forgetting --coreset-ratio 0.1 --mis-ratio 0.3
```