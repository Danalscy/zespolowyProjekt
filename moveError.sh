#!/bin/bash
mkdir "errorData"
for d in $(find Fnt -mindepth 1 -maxdepth 1 -type d)
do
  #Do something, the directory is accessible with $d:
  dire="$(basename "$d")"
  echo "$d"
  #echo $(basename "$d")
  mkdir "errorData/$dire"

  for file in $(ls $d | sort -R | tail -100)
  do
    cp $d/$file errorData/$dire/$file
  done

done 
