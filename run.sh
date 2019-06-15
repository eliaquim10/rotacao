unzip train.rotfaces.zip
unzip test.rotfaces.zip
python util_rotacao.py train.truth.csv rotfaces
python CIFAR.py rotfaces test truth.csv
python girar.py test truth.csv predition
zip predition.zip predition/*