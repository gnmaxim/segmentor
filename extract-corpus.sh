#!/usr/bin/env bash
# Author: Maxim Gaina


TOPDIR=$PWD
FOLDER=0
CONFIG=0

DATA=rawdata
EXTRACTED_FOLDER=timit

WAV=".WAV"
WAVDIR=tmp
NEWWAVDIR=wav
PHNDIR=phn
CSVDIR=csv
# SMILECONF="$(dirname $(which SMILExtract))"


function usage()
{
    echo -e "Usage:"
    echo -e "\n\t-h, --help\n\t\tPrint this message"
    echo -e "\n\t-c, --extract-corpus <FOLDER_PATH>"
    echo -e "\t\textract .WAV files and phonetic segmentation from corpus, requires corpus path as argument"
    echo -e "\n\t-f, --extract-features <OPENSMILE_CONFIGURATION_FILE>"
    echo -e "\t\textract features from .WAV files with SMILExtract, requires openSmile configuration file"
    echo -e "\n\t\tCan be used with --extract-corpus or -c, or it will assume that corpus extraction was performed in the past.\n"
}


TEMP=`getopt -o hc:f: --long help,extract-corpus:,extract-features: -n 'extract_timit.sh' -- "$@"`
eval set -- "$TEMP"

while true;
do
    case "$1" in
        -h|--help) usage ; shift ;;
        -c|--extract-corpus)
            case "$2" in
                "") shift 2 ;;
                *) FOLDER=$2 ; shift 2 ;;
            esac ;;
        -f|--extract-features)
            case "$2" in
                "") shift 2 ;;
                *) CONFIG=$2 ; shift 2 ;;
            esac ;;
        --) shift ; break ;;
         *) echo "Internal error!" ; exit 1 ;;
    esac
done


cd $DATA
cd $EXTRACTED_FOLDER

if [ $FOLDER != 0 ]; then
    TARGET=$(basename $FOLDER)
    rm -rf $TARGET
    mkdir $TARGET
    cd $TARGET

    # Extracting audio files
    mkdir $WAVDIR
    echo "Extracting all .WAVs..."
    find $FOLDER -name "*.WAV" |
    while IFS= read -r NAME;
    do
        # -v verbose
        cp "$NAME" "$WAVDIR/${NAME//\//_}";
    done
    echo "Extracted $(ls $WAVDIR -1 | wc -l) .WAVs.";
    #read -n1 -r -p "Press any key to continue and erase extra .WAF's header content..." key


    # Erasing extra content from .wav header, so files can be processed by opensmile
    mkdir $NEWWAVDIR
    for filename in $WAVDIR/*;
    do
        ffmpeg -y -i $filename -map_metadata -1 -c:v copy -c:a copy $NEWWAVDIR/${filename#$WAVDIR/};
    done
    rm -rf $WAVDIR
    echo "Extra information deleted from all .WAVs."
    # read -n1 -r -p "Press any key to continue and extract corpus phones..." key


    # Extracting phone segmentation
    mkdir $PHNDIR
    echo "Extracting all .PHNs..."
    find $FOLDER -name "*.PHN" |
    while IFS= read -r NAME;
    do
        cp "$NAME" "$PHNDIR/${NAME//\//_}";
    done
    echo "Extracted $(ls $PHNDIR -1 | wc -l) .PHNs."
    #read -n1 -r -p "Press any key to continue and extract needed .WAF's features..." key
fi


if [ $CONFIG != 0 ]; then
    # Extracting features with SMILExtract, assuming it's added to PATH
    mkdir $CSVDIR

    echo "Extracting features into CSVs..."
    for filename in $NEWWAVDIR/*;
    do
        f=${filename%$WAV}
        SMILExtract -C $CONFIG -I $filename -O $CSVDIR/${f#$NEWWAVDIR}.csv;
    done

    echo "Created new $(ls $CSVDIR -1 | wc -l) .CSV feature files."
fi
