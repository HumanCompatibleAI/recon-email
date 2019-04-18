import argparse
import jinja2
import markdown
import re

from make_emails import SpreadsheetReader, md_to_html

COLUMN_NAMES = [
    'Date', 'Added by', 'Authors/Speaker/Hosts/Participants/People involved',
    'Item', 'Type', 'Conference/Journal/Venue/Details', 'Status', 'Grant',
    'Link', 'Summary for CHAI newsletter and/or website', 'Notes',
    'Photos/Media', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
    '', '', '', ''
]

class CHAISpreadsheetReader(SpreadsheetReader):
    """An implementation of SpreadsheetReader for the CHAI Newsletter.

    See SpreadsheetReader for details on each of the methods.
    """

    def __init__(self):
        super(CHAISpreadsheetReader, self).__init__()

    def check_row(self, row):
        pass

    def make_entry(self, row):
        """An entry has the same form as a row: it is a dictionary that assigns
        a value to every column in COLUMN_NAMES.
        """
        return row

    def check_entry(self, entry):
        pass

##########
# Output #
##########

TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <style>
        body {
            font-family: sans-serif;
            font-size: 10pt;
        }
    </style>
</head>
<body>
<div class="container">
{{content}}
</div>
</body>
</html>
"""

def write_output(filename, entries):
    html = ''.join([get_html(e) for e in entries])
    with open(filename, 'w') as out:
        out.write(jinja2.Template(TEMPLATE).render(content=html))

def get_html(entry):
    title, link = entry['Item'], entry['Link']
    if link != '':
        m = re.match(r'^<a href="(\S+)" target="_blank">(.*)</a>$', link)
        if m is not None:
            title = '[{}]({})'.format(title, m.group(1))

    title, author, venue, summary = map(
        md_to_html, [
            title,
            entry['Authors/Speaker/Hosts/Participants/People involved'],
            entry['Conference/Journal/Venue/Details'],
            entry['Summary for CHAI newsletter and/or website']
        ])

    def maybe_format(s, format_str):
        return s if s == '' else format_str.format(s)

    author = maybe_format(author, ' <i>({0})</i>')
    # Not currently using the venue
    summary = maybe_format(summary, ': {0}')
    template = '<p>{0}{1}{2}</p>'
    return template.format(title, author, summary)


def md_to_html(md):
    result = markdown.markdown(str(md), output_format='html5')
    result = result.replace('\n', '</p><p>')
    return result[3:-4]


#################
# Program entry #
#################


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='data/Sheet1.html',
                        help='HTML export of entries from CHAI outputs and news spreadsheet.')
    parser.add_argument('-o', '--output', type=str, default='data/email.html',
                        help='Public output file name. Defaults to email.html.')
    return parser.parse_args(args)


def main():
    args = parse_args()
    entries = CHAISpreadsheetReader().get_entries(
        args.input, 1, 3, COLUMN_NAMES)
    write_output(args.output, entries)


if __name__ == '__main__':
    main()
