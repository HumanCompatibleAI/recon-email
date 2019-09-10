#!/bin/bash

cd data
rm *.html resources/sheet.css
mv ~/Downloads/Reconnaissance.zip .
unzip Reconnaissance.zip
cd ..
python make_emails.py -n $1
open data/public_email.html data/email.html
