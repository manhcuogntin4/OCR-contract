#!/bin/bash
cnt=1
dir_path=$1
for file in `ls -v  $dir_path/*.png`
do
  mv "$file" $dir_path/$cnt.bin.png
  let cnt=cnt+1
done
