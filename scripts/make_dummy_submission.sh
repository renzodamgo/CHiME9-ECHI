#!/bin/bash

#
# TEMPORARY - FOR TESTING
#
# Make a dummy submission.
#
# Generate a submission to allow metrics to be tested.  Uses sox to
# generate 16 kHz and single channel versions of the signals. Signals
# are duplicated to make a copy for each speaker stream.

# /Volumes/ECHI1/chime9_echi/aria/dev/
#  /Volumes/ECHI1/chime9_echi/ha/dev/

ECHI_ROOT=/Volumes/ECHI1/chime9_echi
ARIA_ROOT=${ECHI_ROOT}/aria/dev
HA_ROOT=${ECHI_ROOT}/ha/dev

SUBMISSION=/Volumes/ECHI1/echi_submission

mkdir -p $SUBMISSION

for name in $(ls ${ARIA_ROOT}); do
    echo $name
    sox ${ARIA_ROOT}/$name -r 16k ${SUBMISSION}/$name remix 1,2,3,4,5,6,7
    for pos in pos1 pos2 pos3 pos4; do
        posname=$(echo $name | sed s/.wav/.$pos.wav/g)
        ln -s ${SUBMISSION}/$name ${SUBMISSION}/$posname
    done
done

for name in $(ls ${HA_ROOT} | grep front); do
    echo $name
    sox ${HA_ROOT}/$name -r 16k ${SUBMISSION}/$name remix 1,2
    for pos in pos1 pos2 pos3 pos4; do
        posname=$(echo $name | sed s/.wav/.$pos.wav/g)
        ln -s ${SUBMISSION}/$name ${SUBMISSION}/$posname
    done
done
