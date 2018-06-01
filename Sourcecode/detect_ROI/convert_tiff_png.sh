#!/bin/bash
for file in *.tif
do
  echo $1
  convert "$file" "$file"-%d.png
done
