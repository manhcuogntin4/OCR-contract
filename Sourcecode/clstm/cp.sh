#!/bin/bash
dhome=./dataset_trainning
mkdir Annotations
mkdir Images
mkdir ImageSets
mkdir data
cp $dhome/*.png Images
cp $dhome/*.xml Annotations
ls Annotations/ -m | sed s/\\s/\\n/g | sed s/.xml//g | sed s/,//g > ImageSets/train.txt
mv Annotations data
mv Images data
mv ImageSets data
