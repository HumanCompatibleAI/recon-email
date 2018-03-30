#!/bin/bash

cd data
rm *.html
mv ~/Downloads/Reconnaissance.zip .
unzip Reconnaissance.zip
cd ..
source activate web_scraping
python make-emails.py
source deactivate
open data/email.html data/public_email.html
