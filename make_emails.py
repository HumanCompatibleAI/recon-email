import argparse
import csv
import jinja2
import markdown
import re
import sys

from abc import ABC, abstractmethod
from bs4 import BeautifulSoup


################
# Reading data #
################

class SpreadsheetReader(ABC):
    """Parses a Google Sheet that has been downloaded as an html file.

    Known issue: Can't handle spreadsheets with frozen columns. (It can handle
    frozen rows.)
    """
    def __init__(self):
        super(SpreadsheetReader, self).__init__()

    def get_entries(self, filename, header_index, entry_index, column_names):
        """Parses a Google Sheet that has been downloaded as an html file.

        filename: The name of the HTML file from which to extract entries.
        header_index: The index of the row containing the column names.
        entry_index: The index of the first row containing a potential entry.
        config: Configuration parameters, passed along to user-defined methods.

        Returns a list of Entry objects.

        How to figure out the values of header_index and entry_index:
        Index 0 is the Google Sheets columns (A, B, C, ...). Index 1 is the
        first row with spreadsheet content. If the first N rows are frozen, then
        the index N+1 will be the small grey boxes showing the freezing, and
        content resumes with index N+2.
        """
        with open(filename, 'r', encoding="utf-8") as html_file:
            soup = BeautifulSoup(html_file, 'html.parser')
            rows_soup = soup.find_all('tr')
            header = self.parse_row(rows_soup[header_index])
            assert header == column_names, str(header)

            all_entries = []
            for row_soup in rows_soup[entry_index:]:
                row_dict = dict(zip(header, self.parse_row(row_soup)))
                self.check_row(row_dict)
                entry = self.make_entry(row_dict)
                if entry is not None:
                    self.check_entry(entry)
                    all_entries.append(entry)

            return all_entries

    def parse_row(self, row):
        return [self.get_content(cell) for cell in row.children][1:]

    def get_content(self, thing):
        return self.get_content_from_list([thing])

    def get_content_from_list(self, lst):
        if lst == []:
            return ''
        elif len(lst) == 1:
            thing = lst[0]
            if thing.name not in ['td', 'div', 'span']:
                return str(thing)
            return self.get_content_from_list(list(thing.children))
        else:
            lst = [thing for thing in lst if thing.name != 'br']
            assert all((thing.name is None for thing in lst)), str(lst)
            return '\n'.join([str(x) for x in lst])

    @abstractmethod
    def check_row(self, row):
        """Raises an error if the row from the spreadsheet is malformed. 

        row: Dictionary mapping column names to column values, representing a
             row in the spreadsheet.
        """
        pass

    @abstractmethod
    def make_entry(self, row):
        """Converts a row from the spreadsheet into an entry.

        row: Dictionary mapping column names to column values, representing a
             row in the spreadsheet.

        Returns: An entry, or None if this row should not be turned into an
        entry. An entry can be any type, as long as it is used consistently.
        """
        pass

    @abstractmethod
    def check_entry(self, entry):
        """Raises an error if the entry is malformed.

        entry: An Entry object (see make_entry).

        This is a separate method from check_row because not all rows are turned
        into entries -- when make_entry returns None, the corresponding rows are
        ignored, and so checks in this method will not be applied.
        """
        pass


# Alignment Newsletter specific configuration


COLUMN_NAMES = [
    'Rec?', 'Category', 'Title', 'Authors', 'Venue', 'Year', 'H/T',
    'Summarizer', 'Email status', 'Public?', 'Summarize?', 'Notes', 'Email',
    'Summary', 'My opinion', 'Prerequisites', 'Read more'
]


# List of acceptable values for columns
COLUMN_ENUMS = {
    'Rec?': [
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'
    ],
    'Email status': [
        'Do not send', 'Pending', 'Sent', 'Sent link only', 'N/A'
    ],
    'Public?': [
        'Yes', 'With edits', 'Link only', 'No', ''
    ],

    'Summarize?': [
        'Yes', 'No', '1', '2', '3', '4', '5', ''
    ]
}


HIGHLIGHT_REC = '10'


class ReconSpreadsheetReader(SpreadsheetReader):
    """An implementation of SpreadsheetReader for the Alignment Newsletter.

    See SpreadsheetReader for details on each of the methods.
    """

    def __init__(self, for_database):
        super(ReconSpreadsheetReader, self).__init__()
        self.for_database = for_database

    def check_row(self, row):
        assert row['Email status'] in COLUMN_ENUMS['Email status'], \
            'Invalid email status: {}\nEntry: {}'.format(row['Email status'], row)

    def make_entry(self, row):
        """An entry has the same form as a row: it is a dictionary that assigns
        a value to every column in COLUMN_NAMES.
        """
        if self.for_database and row['Category'] == 'Previous newsletters':
            return None
        elif (not self.for_database) and row['Email status'] != 'Pending':
            return None
        return row

    def check_entry(self, entry):
        for column, column_vals in COLUMN_ENUMS.items():
            assert entry[column] in column_vals, \
                'Column {} has invalid value {} (not one of {})\n\nEntry {}'.format(
                    column, entry[column], column_vals, entry)

        assert entry['Title'] != ''

        summary, opinion = entry['Summary'], entry['My opinion']

        if summary != '' or opinion != '':
            assert entry['Summarizer'] != '', entry

        if entry['Public?'] == 'With edits':
            assert summary != '' or opinion != ''
        if entry['Public?'] == '':
            assert summary == '' and opinion == ''

        if entry['Summarize?'] == 'No':
            assert summary == '' and opinion == ''
        elif entry['Summarize?'] == 'Yes':
            assert summary != ''

        if not self.for_database:
            assert not (summary == '' and opinion != '')
            assert entry['Category'] in CATEGORIES, \
                '{} is not a valid category: {}\n\nEntry {}'.format(
                    entry['Category'], CATEGORIES, entry)


#################
# Category tree #
#################

class Category(object):
    def __init__(self, name, children=[]):
        """Name is a string, children is a list of Category objects."""
        self.name = name
        self.children = children
        self.entries = []

    def is_leaf(self):
        return self.children == []

    def get_leaf_categories(self):
        """Returns a list of leaf category names (strings)."""
        if self.children == []:
            return [self.name]
        return flatten([child.get_leaf_categories() for child in self.children])

    def clone(self):
        return Category(self.name, [child.clone() for child in self.children])


def flatten(lst_of_lsts):
    """Converts a list of lists of X into a list of X."""
    result = []
    for lst in lst_of_lsts:
        result += lst
    return result


CATEGORY_TREE = Category('All', [
    Category('Previous newsletters'),
    Category('Technical AI alignment', [
        Category('Embedded agency sequence'),
        Category('Iterated amplification sequence'),
        Category('Value learning sequence'),
        Category('Fixed point sequence'),
        Category('Summary: Inverse Reinforcement Learning'),
        Category('Problems'),
        Category('Technical agendas and prioritization'),
        Category('Iterated amplification'),
        Category('Scalable oversight'),
        Category('Mesa optimization'),
        Category('Agent foundations'),
        Category('Learning human intent'),
        Category('Reward learning theory'),
        Category('Preventing bad behavior'),
        Category('Handling groups of agents'),
        Category('Game theory'),
        Category('Philosophical deliberation'),
        Category('Interpretability'),
        Category('Adversarial examples'),
        Category('Verification'),
        Category('Robustness'),
        Category('Uncertainty'),
        Category('Forecasting'),
        Category('Critiques (Alignment)'),
        Category('Field building'),
        Category('Miscellaneous (Alignment)'),
    ]),
    Category('Near-term concerns', [
        Category('Fairness and bias'),
        Category('Privacy and security'),
        Category('Machine ethics'),
    ]),
    Category('AI strategy and policy'),
    Category('Malicious use of AI'),
    Category('Other progress in AI', [
        Category('Exploration'),
        Category('Reinforcement learning'),
        Category('Multiagent RL'),
        Category('Deep learning'),
        Category('Meta learning'),
        Category('Unsupervised learning'),
        Category('Hierarchical RL'),
        Category('Applications'),
        Category('Machine learning'),
        Category('AGI theory'),
        Category('Critiques (AI)'),
        Category('Miscellaneous (AI)'),
    ]),
    Category('News'),
])

CATEGORIES = CATEGORY_TREE.get_leaf_categories()


##############
# Processing #
##############

# Columns that can be put into the public version in the "Link only" setting.
NON_SENSITIVE_COLUMNS = [
    'Rec?',
    'Category',
    'Title',
    'Authors',
    'Venue',
    'Year',
    'H/T',
    #'Summarizer',
    'Email status',
    'Public?',
    #'Notes',
    'Email',
    #'Summary',
    #'My opinion',
    #'Prerequisites',
    #'Read more'
],

PUBLICITY_COLUMNS_MAP = {
    'Yes': COLUMN_NAMES,
    'Link only': NON_SENSITIVE_COLUMNS,
    '': NON_SENSITIVE_COLUMNS,
    'With edits': COLUMN_NAMES,
}

def get_public_entries(entries):
    result = []
    for e in entries:
        if e['Public?'] == 'No':
            continue
        elif e['Public?'] not in PUBLICITY_COLUMNS_MAP:
            raise ValueError('Invalid "Public?" value: {}'.format(e['Public?']))
        else:
            e2 = { col:'' for col in COLUMN_NAMES }
            for col in PUBLICITY_COLUMNS_MAP[e['Public?']]:
                e2[col] = e[col]
            result.append(e2)
    return result


def process(entries, tree):
    """Modifies the category tree to add in entries at the appropriate places,
    and prune irrelevant parts of the tree.
    """
    def pick_entries(node):
        """Chooses entries associated with this node (which must be a leaf).

        Ignores highlighted entries, since those are not categorized.
        """
        return [e for e in entries if e['Category'] == node.name and e['Rec?'] != HIGHLIGHT_REC]

    def loop(node):
        if node.children == []:
            node.entries = pick_entries(node)
            node.is_used = len(node.entries) > 0
            return node.is_used

        children_results = [loop(child) for child in node.children]
        node.is_used = any(children_results)
        return node.is_used

    return loop(tree)


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
<p>Audio version <a href="http://alignment-newsletter.libsyn.com/alignment-newsletter-{{number}}">here</a> (may not be up yet).</p>
<h2>Highlights</h2>
<div class="container">
{{content}}
</div>
</body>
</html>
"""

def write_output(filename, entries, database, tree, public, number):
    def loop(node, depth):
        if not node.is_used:
            return ''

        html = '<h{0}>{1}</h{0}>'.format(depth, node.name)
        if not node.is_leaf():
            return html + '<br/>' + ''.join([loop(c, depth+2) for c in node.children])

        entries_html = [get_html(entry, database, public, False) for entry in node.entries]
        html += ''.join(entries_html)
        return html

    highlights = [entry for entry in entries if entry['Rec?'] == HIGHLIGHT_REC]
    html = ''.join([get_html(e, database, public, True) for e in highlights])
    html += ''.join([loop(child, 1) for child in tree.children])
    with open(filename, 'w') as out:
        out.write(jinja2.Template(TEMPLATE).render(content=html, number=number))

def get_html(entry, database, public, highlight_section):
    # Highlighted entries should only go in the highlights section.
    assert highlight_section == (entry['Rec?'] == HIGHLIGHT_REC)
    
    def lookup(link_text, database_key):
        database_key = database_key.lower()
        if database_key not in database:
            raise ValueError('Unknown post/paper (entry not found in database): ' + database_key)
        link, email = database[database_key]
        return '[{0}]({1}) ({2})'.format(link_text, link, email)

    def format_authors(authors_text):
        # No author field
        if authors_text.strip() == '':
            return ''
        # All authors contributed equally
        elif ' and ' in authors_text:
            return authors_text.strip()

        authors = [author.strip() for author in authors_text.split(',')]

        # Single author
        if len(authors) == 1:
            return authors_text.strip()
        # Multiple authors; single first author
        elif authors[0][-1] != '*':
            return authors[0] + ' et al'
        # Multiple first authors
        else:
            first_authors = [author for author in authors if author[-1] == '*']
            # These should be the first n authors, with n > 1, but also not all authors
            assert 1 < len(first_authors) < len(authors)
            assert first_authors[-1] == authors[len(first_authors) - 1]
            # Remove asterisks
            first_authors = [author[:-1] for author in first_authors]
            return ', '.join(first_authors) + ' et al'

    def spreadsheet_text_to_html(text):
        # Handle tags <@ @>(@ @) and <@ @>
        text = re.sub(r"<@([^@]*)@>\(@([^@]*)@\)", lambda m: lookup(m.group(1), m.group(2)), text)
        text = re.sub(r"<@([^@]*)@>", lambda m: lookup(m.group(1), m.group(1)), text)
        result = md_to_html(text)
        return result[3:-4]  # Strip off the starting <p> and ending </p>

    title, author, summarizer_name, hattip, summary, opinion, prereqs, read_more = map(
        spreadsheet_text_to_html, [
            entry['Title'],
            format_authors(entry['Authors']),
            entry['Summarizer'],
            entry['H/T'],
            entry['Summary'],
            entry['My opinion'],
            entry['Prerequisites'],
            entry['Read more']
        ])

    def maybe_format(s, format_str):
        return s if s == '' else format_str.format(s)

    if entry['Public?'] != 'Yes':
        if public:
            # In anything except "With edits", summary + opinion should be empty
            summary = maybe_format(summary, '<b><i><u>{0}</u></i></b>')
            opinion = maybe_format(opinion, '<b><i><u>{0}</u></i></b>')
        else:
            summary = maybe_format(summary, '<i>{0}</i>')
            opinion = maybe_format(opinion, '<i>{0}</i>')

    author = maybe_format(author, ' <i>({0})</i>')
    hattip = maybe_format(hattip, ' (H/T {0})')
    summarizer = maybe_format(summarizer_name, ' (summarized by {0})')
    summary = maybe_format(summary, ': {0}')
    if opinion != '':
        opinion = "</p><p><b>{0}'s opinion:</b> {1}".format(summarizer_name, opinion)
    prereqs = maybe_format(prereqs, '</p><p><b>Prerequisities:</b> {0}')
    read_more = maybe_format(read_more, '</p><p><b>Read more:</b> {0}')
    template = '<p>{0}{1}{2}{3}{4}{5}{6}{7}</p>'
    return template.format(title, author, summarizer, hattip, summary, opinion, prereqs, read_more)


def md_to_html(md):
    result = markdown.markdown(str(md), output_format='html5')
    result = result.replace('\n', '</p><p>')
    return result


def make_database(entries):
    result = {}
    for e in entries:
        m = re.match(r'^<a href="(\S+)" target="_blank">(.*)</a>$', e['Title'])
        if m is not None:
            result[m.group(2).lower()] = (m.group(1), e['Email'])
    return result


#################
# Program entry #
#################


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='data/New entries.html',
                        help='HTML export of new entries from reconnaissance spreadsheet.')
    parser.add_argument('-d', '--database', type=str, default='data/Summarized entries.html',
                        help='HTML export of existing summaries from reconnaissance spreadsheet.')
    parser.add_argument('-c', '--chai_output', type=str, default='data/email.html',
                        help='Output file name. Defaults to email.html.')
    parser.add_argument('-p', '--public_output', type=str, default='data/public_email.html',
                        help='Public output file name. Defaults to public_email.html.')
    parser.add_argument('-n', '--number', type=int, required=True,
                        help='Issue number for the newsletter')
    return parser.parse_args(args)


def main():
    args = parse_args()
    # See comments in SpreadsheetReader to understand the numeric arguments.
    entries = ReconSpreadsheetReader(False).get_entries(
        args.input, 1, 3, COLUMN_NAMES)
    database_entries = ReconSpreadsheetReader(True).get_entries(
        args.database, 2, 4, COLUMN_NAMES)
    database = make_database(database_entries)

    chai_tree = CATEGORY_TREE.clone()
    process(entries, chai_tree)
    write_output(args.chai_output, entries, database, chai_tree, False, args.number)

    public_entries = get_public_entries(entries)
    public_tree = CATEGORY_TREE.clone()
    process(public_entries, public_tree)
    write_output(args.public_output, public_entries, database, public_tree, True, args.number)

if __name__ == '__main__':
    main()
