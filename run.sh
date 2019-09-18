#!/bin/bash

cd /mnt/c/users/georg\ arndt/pycharmprojects/an-converter/data
rm *.html resources/sheet.css
mv /mnt/c/users/georg\ arndt/downloads/Reconnaissance.zip .
unzip Reconnaissance.zip
cd ..

read -p "AN#?  : " num

python3 make_emails.py -n $num

cd data

explorer.exe .
