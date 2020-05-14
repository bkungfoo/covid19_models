#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
if [ ! -d "$DIR/../COVID-19" ]
then
  (cd $DIR/.. && git clone https://github.com/CSSEGISandData/COVID-19)
else
  (cd $DIR/../COVID-19 && git pull)
fi

mkdir -p data
cp census/* data/

if [ -z "$1" ]
then
  python preprocess_data.py
else
  python preprocess_data.py --start_date $1 --end_date $2
fi