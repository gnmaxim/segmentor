#!/usr/bin/env bash
# ffmpeg
# SMILExtract

TOPDIR=$PWD

TIMIT=/home/maxim/Desktop/prominator/timit/TIMIT/
echo "TIMIT directory: $TIMIT"

DATA=data
EXTRACTED_TIMIT=timit

WAVDIR=wav
PHNDIR=phn
CSVDIR=csv
SMILECONF="$(dirname $(which SMILExtract))"
CONFIG=$SMILECONF/config/prosodyAcf.conf


cd $DATA
cd $EXTRACTED_TIMIT
# Extracting audio files
rm -rf $WAVDIR
mkdir $WAVDIR
echo "Extracting all .WAVs..."
find $TIMIT -name "*.WAV" | 
while IFS= read -r NAME;
do
    # -v verbose
    cp "$NAME" "$WAVDIR/${NAME//\//_}";
done
echo "Extracted $(ls $WAVDIR -1 | wc -l) .WAVs."
read -n1 -r -p "Press any key to continue and erase extra .WAF's header content..." key
#rename WAV wav wav/*.WAV

# Erasing extra content from .wav header, so files can be processed by opensmile
for filename in $WAVDIR/*;
do
    ffmpeg -y -i $filename -map_metadata -1 -c:v copy -c:a copy $filename;
done
echo "Extra information deleted from all .WAVs."
read -n1 -r -p "Press any key to continue and extract corpus phones..." key

# Extracting phone segmentation
rm -rf $PHNDIR
mkdir $PHNDIR
echo "Extracting all .PHNs..."
find $TIMIT -name "*.PHN" | 
while IFS= read -r NAME; 
do 
    cp "$NAME" "$PHNDIR/${NAME//\//_}"; 
done
echo "Extracted $(ls $PHNDIR -1 | wc -l) .PHNs."
read -n1 -r -p "Press any key to continue and extract needed .WAF's features..." key

# Extracting features with SMILExtract, assuming it's added to PATH
rm -rf $CSVDIR
mkdir $CSVDIR
echo "Extracting features into CSVs..."
for filename in $WAVDIR/*;
do
    SMILExtract -C $CONFIG -I $filename -O $CSVDIR/${filename#$WAVDIR}.csv;
done
echo "Created new $(ls $CSVDIR -1 | wc -l) .CSV feature files."

