#!/bin/bash

cd data
rm *.html resources/sheet.css
mv ~/Downloads/Reconnaissance.zip .
unzip Reconnaissance.zip
cd ..
source activate web_scraping
python make_emails.py
source deactivate
open data/public_email.html data/email.html
