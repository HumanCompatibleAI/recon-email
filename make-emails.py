import argparse
import csv
import jinja2
import markdown
import sys
from bs4 import BeautifulSoup

################
# Reading data #
################

COLUMN_NAMES = ['Rec?', 'Title', 'Authors', 'Venue', 'Year', 'H/T', 'Email Date', 'Summary', 'My opinion', 'Prerequisites', 'Read more', 'Email status', 'Public?', 'Category']

def get_entries(filename):
    """Reads and parses the reconnaissance csv.

    Returns a list of Entry objects.
    """
    with open(filename, 'r') as html_file:
        soup = BeautifulSoup(html_file, 'html.parser')
        rows = soup.find_all('tr')
        # rows[0] is the name of the columns (A, B, C, ...) so ignore it
        header = parse_row(rows[1])
        # rows[2] shows how the top row is frozen, so ignore it
        assert header == COLUMN_NAMES, str(header)
        row_dicts = [dict(zip(header, parse_row(row))) for row in rows[3:]]
        for row in row_dicts:
            assert row['Email status'] in ['Do not send', 'Pending', 'Sent', 'Sent link only'], str(row)
        return [make_entry(row) for row in row_dicts if row['Email status'] == 'Pending']

def parse_row(row):
    return [get_content(cell) for cell in row.children][1:]

def get_content(thing):
    return get_content_from_list([thing])

def get_content_from_list(lst):
    if lst == []:
        return ''
    elif len(lst) == 1:
        thing = lst[0]
        if thing.name not in ['td', 'div']:
            return str(thing)
        return get_content_from_list(list(thing.children))
    else:
        lst = [thing for thing in lst if thing.name != 'br']
        assert all((thing.name is None for thing in lst))
        return '\n'.join([str(x) for x in lst])

def make_entry(row):
    rec = row['Rec?']
    title, author, hattip = row['Title'], row['Authors'], row['H/T']
    summary, opinion = row['Summary'], row['My opinion']
    prereqs, read_more = row['Prerequisites'], row['Read more']
    is_public = row['Public?']
    category = row['Category']
    return Entry(rec, title, author, hattip, summary, opinion, prereqs, read_more, is_public, category)


###################
# Data structures #
###################

IS_PUBLIC_OPTIONS = ['Yes', 'With edits', 'Link only', 'No', '']

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
    Category('Technical AI alignment', [
        Category('Problems'),
        Category('Technical agendas and prioritization'),
        Category('Iterated distillation and amplification'),
        Category('Scalable oversight'),
        Category('Agent foundations'),
        Category('Learning human intent'),
        Category('Reward learning theory'),
        Category('Handling groups of agents'),
        Category('Interpretability'),
        Category('Forecasting'),
        Category('Critiques (Alignment)'),
        Category('Miscellaneous (Alignment)'),
    ]),
    Category('Near-term concerns', [
        Category('Adversarial examples'),
        Category('Fairness and bias'),
        Category('Privacy and security'),
    ]),
    Category('AI strategy and policy'),
    Category('Malicious use of AI'),
    Category('AI capabilities', [
        Category('Reinforcement learning'),
        Category('Deep learning'),
        Category('Meta learning'),
        Category('Adversarial training'),
        Category('Machine learning'),
        Category('AGI theory'),
        Category('Critiques (Capabilities)'),
        Category('Miscellaneous (Capabilities)'),
    ]),
    Category('News'),
])

CATEGORIES = CATEGORY_TREE.get_leaf_categories()

class Entry(object):
    def __init__(self, rec, title, author, hattip, summary, opinion, prereqs, read_more, is_public, category, review=False):
        assert 1 <= int(rec) <= 5
        self.rec = int(rec)
        assert title != ''
        self.title = title
        self.author = author
        self.hattip = hattip
        if (opinion == '') != (summary == ''):
            assert opinion == ''
            print('Warning: {0} has a summary but no "My opinion" section'.format(title))
        self.is_only_link = (summary == '')
        self.summary = summary
        self.opinion = opinion
        self.prereqs = prereqs
        self.read_more = read_more
        assert is_public in IS_PUBLIC_OPTIONS
        if is_public == 'With edits':
            assert summary != ''
        if is_public == '':
            assert summary == ''
        self.is_public = is_public
        assert category in CATEGORIES, category
        self.category = category
        self.review = review

    def get_html(self, highlight_section=False):
        title, author, hattip, summary, opinion, prereqs, read_more = map(
            md_to_html, [self.title, self.author, self.hattip, self.summary, self.opinion, self.prereqs, self.read_more])
        if self.is_public != 'Yes':
            if summary != '':
                summary = '<i>{0}</i>'.format(summary)
            if opinion != '':
                opinion = '<i>{0}</i>'.format(opinion)

        if self.rec == 5:
            title = '<b>{0}</b>'.format(title)
        if author != '':
            author = ' <i>({0})</i>'.format(author)
        if hattip != '':
            hattip = ' (H/T {0})'.format(hattip)
        if opinion != '':
            opinion = '<br/><b>My opinion:</b> {0}'.format(opinion)
        if prereqs != '':
            prereqs = '<br/><b>Prerequisities:</b> {0}'.format(prereqs)
        if read_more != '':
            read_more = '<br/><b>Read more:</b> {0}'.format(read_more)

        if self.is_only_link:
            return '{0}{1}{2}'.format(title, author, hattip)
        if self.rec == 5 and not highlight_section:
            return '{0}{1}{2}: Summarized in the highlights!'.format(title, author, hattip)

        template = '{0}{1}{2}: {3}{4}{5}{6}'
        if self.review:
            template = '{0}{1}{2}: <b><i><u>{3}</u></i></b>{4}{5}{6}'
        return template.format(title, author, hattip, summary, opinion, prereqs, read_more)

def md_to_html(md):
    result = markdown.markdown(str(md), output_format='html5')
    result = result.replace('\n', '<br/>')
    return result[3:-4]  # Strip off the starting <p> and ending </p>


##############
# Processing #
##############

def get_public_entries(entries):
    result = []
    for e in entries:
        if e.is_public == 'No':
            continue
        if e.is_public in ['Link only', '']:
            e2 = Entry(e.rec, e.title, e.author, e.hattip, '', '', '', '', e.is_public, e.category)
        elif e.is_public == 'Yes':
            e2 = e
        elif e.is_public == 'With edits':
            e2 = Entry(e.rec, e.title, e.author, e.hattip, e.summary, e.opinion, e.prereqs,
                       e.read_more, e.is_public, e.category, review=True)
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

def write_output(filename, entries, tree):
    def highlight_html(entry):
        return entry.get_html(highlight_section=True)

    def loop(node, depth):
        if not node.is_used:
            return ''

        html = '<h{0}>{1}</h{0}>'.format(depth, node.name)
        if not node.is_leaf():
            return html + '<br/>' + ''.join([loop(c, depth+2) for c in node.children])

        entries_html = [entry.get_html() for entry in node.entries]
        html += '<p>' + '</p><p>'.join(entries_html) + '</p><br/>'
        return html

    highlights = [entry for entry in entries if entry.rec == 5]
    html = '<p>' + '</p><p>'.join([highlight_html(e) for e in highlights]) + '</p><br/>'
    html += ''.join([loop(child, 1) for child in tree.children])
    with open(filename, 'w') as out:
        out.write(jinja2.Template(TEMPLATE).render(content=html))

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='data/New entries.html',
                        help='HTML export from reconnaissance spreadsheet.')
    parser.add_argument('-c', '--chai_output', type=str, default='data/email.html',
                        help='Output file name. Defaults to email.html.')
    parser.add_argument('-p', '--public_output', type=str, default='data/public_email.html',
                        help='Public output file name. Defaults to public_email.html.')
    return parser.parse_args(args)

def main():
    args = parse_args()
    entries = get_entries(args.input)
    chai_tree = CATEGORY_TREE.clone()
    process(entries, chai_tree)
    write_output(args.chai_output, entries, chai_tree)
    public_entries = get_public_entries(entries)
    public_tree = CATEGORY_TREE.clone()
    process(public_entries, public_tree)
    write_output(args.public_output, public_entries, public_tree)

if __name__ == '__main__':
    main()
