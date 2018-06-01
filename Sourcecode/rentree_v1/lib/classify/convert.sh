#!/bin/bash
cnt=1
for file in *.pdf
do
  echo $file
  pdftoppm -png -rx 300 -ry 300 -png "$file" "$file"
  let cnt=cnt+1
done
