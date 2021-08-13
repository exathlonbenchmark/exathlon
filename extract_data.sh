#!/bin/bash

unzip "./data/raw/ground_truth.zip"
rm "./data/raw/ground_truth.zip"
for (( i=1; i<=10; i++ )); do
  DIR="./data/raw/app$i"
  unzip "$DIR/*.zip" -d $DIR
  find $DIR -maxdepth 1 -type f -name "*.zip" -delete
  for subdir in $(find $DIR -mindepth 1 -type d); do
    ZIP_FILE_PATH=$(find $subdir -iname \*.zip)
    ZIP_FILE_NAME=${ZIP_FILE_PATH##*/}
    # zip -s0 caused some issues with some files so we used cat instead
    ZIP_FILE_PATH_NO_EXT=${ZIP_FILE_PATH%.*}
    cat $ZIP_FILE_PATH_NO_EXT.z0* $ZIP_FILE_PATH > $DIR/$ZIP_FILE_NAME
    unzip $DIR/$ZIP_FILE_NAME -d $DIR
    rm $DIR/$ZIP_FILE_NAME
    rm -rf $subdir
  done
done
