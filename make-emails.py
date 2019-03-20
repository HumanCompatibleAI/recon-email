import argparse
import csv
import jinja2
import markdown
import re
import sys
from bs4 import BeautifulSoup

################
# Reading data #
################

COLUMN_NAMES = ['Rec?', 'Category', 'Title', 'Authors', 'Venue', 'Year', 'H/T', 'Summarizer', 'Email status', 'Public?', 'Notes', 'Email', 'Summary', 'My opinion', 'Prerequisites', 'Read more']

def get_entries(filename, header_index, for_database):
    """Reads and parses the reconnaissance csv.

    filename: The name of the HTML file from which to extract entries
    header_index: The index of the row containing the column names
    for_database: If True the entries are only for the database, so be a bit lax about input validation. Keep all entries, not just Pending ones, and don't require the category to exist.

    Returns a list of Entry objects.
    """
    with open(filename, 'r') as html_file:
        soup = BeautifulSoup(html_file, 'html.parser')
        rows = soup.find_all('tr')
        header = parse_row(rows[header_index])
        assert header == COLUMN_NAMES, str(header)
        # rows[header_index + 1] shows how the top row is frozen, so ignore it
        row_dicts = [dict(zip(header, parse_row(row))) for row in rows[header_index + 2:]]
        for row in row_dicts:
            assert row['Email status'] in ['Do not send', 'Pending', 'Sent', 'Sent link only', 'N/A'], str(row)

        def use_row(row):
            if for_database:
                return row['Category'] != 'Previous newsletters'
            else:
                return row['Email status'] == 'Pending'

        return [make_entry(row, for_database) for row in row_dicts if use_row(row)]

def parse_row(row):
    return [get_content(cell) for cell in row.children][1:]

def get_content(thing):
    return get_content_from_list([thing])

def get_content_from_list(lst):
    if lst == []:
        return ''
    elif len(lst) == 1:
        thing = lst[0]
        if thing.name not in ['td', 'div', 'span']:
            return str(thing)
        return get_content_from_list(list(thing.children))
    else:
        lst = [thing for thing in lst if thing.name != 'br']
        assert all((thing.name is None for thing in lst)), str(lst)
        return '\n'.join([str(x) for x in lst])

def make_entry(row, for_database):
    rec, category = row['Rec?'], row['Category']
    title, author, hattip = row['Title'], row['Authors'], row['H/T']
    summary, opinion = row['Summary'], row['My opinion']
    prereqs, read_more = row['Prerequisites'], row['Read more']
    summarizer, is_public = row['Summarizer'], row['Public?']
    email = row['Email']
    return Entry(rec, category, title, author, hattip, summary, opinion, prereqs, read_more, summarizer, email, is_public, for_database=for_database)


###################
# Data structures #
###################

HIGHLIGHT_REC = 10
IS_PUBLIC_OPTIONS = ['Yes', 'With edits', 'Link only', 'No', '']
SUMMARIZER_OPTIONS = ['Richard', 'Dan H', '']

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

class Entry(object):
    def __init__(self, rec, category, title, author, hattip, summary, opinion, prereqs, read_more, summarizer, email, is_public, review=False, for_database=False):
        assert 1 <= int(rec) <= HIGHLIGHT_REC
        self.rec = int(rec)
        assert title != ''
        self.title = title
        self.author = author
        self.hattip = hattip
        if (opinion == '') != (summary == '') and not for_database:
            assert opinion == ''
            print('Warning: {0} has a summary but no "My opinion" section'.format(title))
        self.summary = summary
        self.opinion = opinion
        self.prereqs = prereqs
        self.read_more = read_more
        assert summarizer in SUMMARIZER_OPTIONS
        self.summarizer = summarizer
        self.email = email
        assert is_public in IS_PUBLIC_OPTIONS
        if is_public == 'With edits':
            assert summary != ''
        if is_public == '':
            assert summary == ''
        self.is_public = is_public
        if not for_database:
            assert category in CATEGORIES, category
        self.category = category
        self.review = review

    def get_html(self, database, highlight_section=False):
        make_html = lambda text: self.spreadsheet_text_to_html(text, database)
        title, author, summarizer_name, hattip, summary, opinion, prereqs, read_more = map(
            make_html, [self.title, self.author, self.summarizer, self.hattip, self.summary, self.opinion, self.prereqs, self.read_more])

        def maybe_format(s, format_str):
            return s if s == '' else format_str.format(s)

        if self.review:
            summary = maybe_format(summary, '<b><i><u>{0}</u></i></b>')
            opinion = maybe_format(opinion, '<b><i><u>{0}</u></i></b>')
        elif self.is_public != 'Yes':
            summary = maybe_format(summary, '<i>{0}</i>')
            opinion = maybe_format(opinion, '<i>{0}</i>')

        if self.rec == HIGHLIGHT_REC:
            title = maybe_format(title, '<b>{0}</b>')
        author = maybe_format(author, ' <i>({0})</i>')
        hattip = maybe_format(hattip, ' (H/T {0})')
        summarizer = maybe_format(summarizer_name, ' (summarized by {0})')
        summary = maybe_format(summary, ': {0}')
        if opinion != '':
            opinion = "</p><p><b>{0}'s opinion:</b> {1}".format(
                ('Rohin' if summarizer_name == '' else summarizer_name), opinion)
        prereqs = maybe_format(prereqs, '</p><p><b>Prerequisities:</b> {0}')
        read_more = maybe_format(read_more, '</p><p><b>Read more:</b> {0}')

        if self.rec == HIGHLIGHT_REC and not highlight_section:
            return '<p>{0}{1}{2}: Summarized in the highlights!</p>'.format(title, author, hattip)

        template = '<p>{0}{1}{2}{3}{4}{5}{6}{7}</p>'
        return template.format(title, author, summarizer, hattip, summary, opinion, prereqs, read_more)

    def spreadsheet_text_to_html(self, text, database):
        def lookup(link_text, database_key):
            if database_key not in database:
                print('Unknown post/paper (entry not found in database): ' + database_key)
            link, email = database[database_key]
            return '[{0}]({1}) ({2})'.format(link_text, link, email)

        # Handle tags <@ @>(@ @) and <@ @>
        text = re.sub(r"<@([^@]*)@>\(@([^@]*)@\)", lambda m: lookup(m.group(1), m.group(2)), text)
        text = re.sub(r"<@([^@]*)@>", lambda m: lookup(m.group(1), m.group(1)), text)
        result = md_to_html(text)
        return result[3:-4]  # Strip off the starting <p> and ending </p>


def md_to_html(md):
    result = markdown.markdown(str(md), output_format='html5')
    result = result.replace('\n', '</p><p>')
    return result


def make_database(entries):
    result = {}
    for e in entries:
        m = re.match(r'^<a href="(\S+)" target="_blank">(.*)</a>$', e.title)
        if m is not None:
            result[m.group(2)] = (m.group(1), e.email)
    return result


##############
# Processing #
##############

def get_public_entries(entries):
    result = []
    for e in entries:
        if e.is_public == 'No':
            continue
        if e.is_public in ['Link only', '']:
            e2 = Entry(e.rec, e.category, e.title, e.author, e.hattip, '', '', '', '', '', e.email, e.is_public)
        elif e.is_public == 'Yes':
            e2 = e
        elif e.is_public == 'With edits':
            e2 = Entry(e.rec, e.category, e.title, e.author, e.hattip, e.summary, e.opinion, e.prereqs,
                       e.read_more, e.summarizer, e.email, e.is_public, review=True)
        else:
            raise ValueError('Invalid value of is_public: ' + str(e.is_public))
        result.append(e2)
    return result


def process(entries, tree):
    """Modifies the category tree to add in entries at the appropriate places,
    and prune irrelevant parts of the tree.
    """
    def pick_entries(node):
        """Chooses entries associated with this node (which must be a leaf category)."""
        return [e for e in entries if e.category == node.name]

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
<h2>Highlights</h2>
<div class="container">
{{content}}
</div>
</body>
</html>
"""

def write_output(filename, entries, database, tree):
    def highlight_html(entry):
        return entry.get_html(database, highlight_section=True)

    def loop(node, depth):
        if not node.is_used:
            return ''

        html = '<h{0}>{1}</h{0}>'.format(depth, node.name)
        if not node.is_leaf():
            return html + '<br/>' + ''.join([loop(c, depth+2) for c in node.children])

        entries_html = [entry.get_html(database) for entry in node.entries]
        html += ''.join(entries_html)
        return html

    highlights = [entry for entry in entries if entry.rec == HIGHLIGHT_REC]
    html = ''.join([highlight_html(e) for e in highlights])
    html += ''.join([loop(child, 1) for child in tree.children])
    with open(filename, 'w') as out:
        out.write(jinja2.Template(TEMPLATE).render(content=html))

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
    return parser.parse_args(args)

def main():
    args = parse_args()
    entries = get_entries(args.input, 1, False)
    database = make_database(get_entries(args.database, 2, True))
    chai_tree = CATEGORY_TREE.clone()
    process(entries, chai_tree)
    write_output(args.chai_output, entries, database, chai_tree)
    public_entries = get_public_entries(entries)
    public_tree = CATEGORY_TREE.clone()
    process(public_entries, public_tree)
    write_output(args.public_output, public_entries, database, public_tree)

if __name__ == '__main__':
    main()
