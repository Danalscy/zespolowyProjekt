#!/bin/bash
mkdir "trainingData"
for d in $(find Fnt -mindepth 1 -maxdepth 1 -type d)
do
  #Do something, the directory is accessible with $d:
  dire="$(basename "$d")"
  echo "$d"
  #echo $(basename "$d")
  mkdir "trainingData/$dire"

  for file in $(ls $d | sort -n | head -200)
  do
    cp $d/$file trainingData/$dire/$file
  done

done 
