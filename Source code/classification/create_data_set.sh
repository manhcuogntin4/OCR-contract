#!/bin/bash
DATA_ROOT="DATASET"
mkdir $DATA_ROOT/resized
mkdir $DATA_ROOT/resized/merge
for dir in signee nonsignee;
  do
    echo Resize of directory: $dir ;
    case ${dir:0:2} in
      si)
        CLASS=1
        ;;
      no)
        CLASS=2
        ;;
      *)
        echo "Not found"
        exit 1
    esac
    echo Classe : $CLASS ;
    for i in $DATA_ROOT/$dir/*;
      do
        echo $dir/`basename "$i"`,$CLASS,0,0,0,0,0,0 >> $DATA_ROOT/$dir.csv ;
      done
    ./bin/extractRect $DATA_ROOT/$dir.csv --resize_width=224 --full_image --noise_rotation=180 --samples 5 $DATA_ROOT/resized/$dir ;
    cat $DATA_ROOT/resized/$dir/results.csv >> $DATA_ROOT/resized/merge/results.csv
  done
mkdir $DATA_ROOT/resized/merge/1
mkdir $DATA_ROOT/resized/merge/2
cp $DATA_ROOT/resized/signee/1/* $DATA_ROOT/resized/merge/1/
cp $DATA_ROOT/resized/nonsignee/2/* $DATA_ROOT/resized/merge/2/
ln -s $DATA_ROOT/resized/merge results/resized