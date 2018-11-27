# recon-email
Script for automatically creating the Alignment Newsletter email from the database (a Google spreadsheet). Download the database as HTML, and then execute `run.sh`, which should automatically open two tabs in your browser with text that can be copy pasted into an email. One tab contains the private version (sent for review to particular people) and one contains the public version (sent via Mailchimp to all subscribers).

I wrote this script for the sole purpose of working with a spreadsheet that I control. As a result, the script is very specific to the spreadsheet, and I put no thought into reusability, portability etc. Reuse this code at your own risk.
