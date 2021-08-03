#!/bin/bash

unzip "./data/raw/ground_truth.zip"
rm "./data/raw/ground_truth.zip"
for (( i=1; i<=10; i++ )); do
  DIR="./data/raw/app$i"
  unzip "$DIR/*.zip" -d $DIR
  find $DIR -name "*.zip" -type f -delete
  for subdir in $(find $DIR -mindepth 1 -type d); do
    ZIP_FILE_PATH=$(find $subdir -iname \*.zip)
    zip -F $ZIP_FILE_PATH --out $DIR/$(basename $ZIP_FILE_PATH)
    unzip $DIR/$(basename $ZIP_FILE_PATH) -d $DIR
    rm -rf $subdir
  done
done