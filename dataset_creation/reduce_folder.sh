#!/bin/bash
cd ./data/img_align_celeba

for FILE in *.jpg
do 
	echo $FILE
	convert $FILE -resize 180x220 ../celeba_180x220/$FILE
done

cd ../celeba_180x220

for FILE in *.jpg
do
	echo $FILE
	convert $FILE -filter Catrom -resize 25% ../celeba_55x45/$FILE
done



 #convert ./img_align_celeba/00000*.jpg -filter Catrom -resize 25% ./img_align_celeba_resized_25/out%d.jpg