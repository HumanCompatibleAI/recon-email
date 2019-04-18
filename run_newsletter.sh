#!/bin/bash

cd data
rm *.html resources/sheet.css
mv ~/Downloads/CHAI\ outputs\ and\ news.zip .
unzip CHAI\ outputs\ and\ news.zip
cd ..
source activate web_scraping
python chai_newsletter.py
source deactivate
open data/email.html
