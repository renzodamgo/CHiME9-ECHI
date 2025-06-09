#!/bin/bash

#
# TEMPORARY - FOR TESTING
#
# Make a dummy submission.
#
# Generate a submission to allow metrics to be tested.  Uses sox to
# generate 16 kHz and single channel versions of the signals. Signals
# are duplicated to make a copy for each speaker stream.

if [ -z "$1" ]; then
    echo "Usage: $0 <chime9_echi_dir> <submission_directory>"
    exit 1
fi

ECHI_ROOT=$1
SUBMISSION=$2

ARIA_ROOT=${ECHI_ROOT}/aria/dev
HA_ROOT=${ECHI_ROOT}/ha/dev
REF_DIR=${ECHI_ROOT}/ref/dev

mkdir -p $SUBMISSION

# Make the Aria submission
for name in $(ls ${ARIA_ROOT}); do
    echo $name
    session=$(echo $name | cut -d'.' -f1)
    pids=$(ls ${REF_DIR} | grep P | grep aria | grep $session | cut -d'.' -f3)
    sox ${ARIA_ROOT}/$name -r 16k ${SUBMISSION}/$name remix 1,2,3,4,5,6,7
    echo $pids
    for pid in $pids; do
        pidname=$(echo $name | sed s/.wav/.$pid.wav/g)
        echo $name $pidname
        (
            cd ${SUBMISSION}
            ln -s $name $pidname
        )
    done
done

# Make the HA submission
for name in $(ls ${HA_ROOT} | grep front); do
    echo $name
    session=$(echo $name | cut -d'.' -f1)
    outname=$(echo $name | sed s/_front//g)
    sox ${HA_ROOT}/$name -r 16k ${SUBMISSION}/$outname remix 1,2
    pids=$(ls ${REF_DIR} | grep P | grep ha | grep $session | cut -d'.' -f3)
    echo $pids
    for pid in $pids; do
        pidname=$(echo $outname | sed s/.wav/.$pid.wav/g)
        echo $name $pidname
        (
            cd ${SUBMISSION}
            ln -s $outname $pidname
        )
    done
done
