import re

WHITESPACE = re.compile(r'(?: |\n)+')
CHARACTERS = ['A-z', '0-9', '"', '(', ')', ',', '-', ':', '=', '{', '|', '}']
RESERVED_WORDS = ["TAGS", "BEGIN", "SEQUENCE", "INTEGER", "DATE", "END"]
# TOKENS = ['Range_Seperator', 'ASSIGN', 'LCURLY', 'RCURLY', 'COMMA', 'LPAREN', 'RPAREN', 'TypeRef', 'Identifier', 'Number']
TOKENS = {
    '..': 'Range_Seperator',
    '::=': 'ASSIGN',
    '{': 'LCURLY',
    '}': 'RCURLY',
    ',': 'COMMA',
    '-': 'HYPHEN',
    '|': 'VERTICAL LINE',
    '(': 'LPAREN',
    ')': 'RPAREN',
    "TAGS": 'RESERVED_WORD',
    "BEGIN": 'RESERVED_WORD',
    "SEQUENCE": 'RESERVED_WORD',
    "INTEGER": 'RESERVED_WORD',
    "DATE": 'RESERVED_WORD',
    "END": 'RESERVED_WORD',
}
REG_TOKENS = {
    '\.\.': 'Range_Seperator',
    '::=': 'ASSIGN',
    '\{': 'LCURLY',
    '\}': 'RCURLY',
    '\,': 'COMMA',
    '\(': 'LPAREN',
    '\)': 'RPAREN',
    "TAGS": 'RESERVED_WORD',
    "BEGIN": 'RESERVED_WORD',
    "SEQUENCE": 'RESERVED_WORD',
    "INTEGER": 'RESERVED_WORD',
    "DATE": 'RESERVED_WORD',
    "END": 'RESERVED_WORD',
}
TOKEN_REGEX = re.compile('|'.join([f'(?:{x})' for x in REG_TOKENS.keys()]))

REGEX_TOKENS = {
    'TypeRef': re.compile(r'[A-Z](?:(?:[A-z]|[0-9])+-?)*'),#(?=,|;|$)'),
    'Identifier': re.compile(r'[a-z](?:(?:[A-z]|[0-9])+-?)*'),#(?=,|;|$)'),
    'Integer': re.compile(r'(?!\=)[0-9]+')#,
    #'RealNumber': re.compile(r'(?!=)[0-9]+\.?[0-9]+')
}

# Reserved Words
# reserved_regex = re.compile(r'|'.join([f'(?! )(?:{x})' for x in RESERVED_WORDS]))
# Tokens

# Lexical Characters
# typeref_regex = re.compile(r'(?! )[A-Z](?:(?:[A-z]|[0-9])+-?)*(?= |,|;)')
# identifier_regex = re.compile(r'(?! )[a-z](?:(?:[A-z]|[0-9])+-?)*(?= |,|;)')
# number_regex = re.compile(r'(?! |=)[0-9]\.?[0-9](?=\.)')
# assignment_regex = re.compile(r'::=')
# range_regex = re.compile(r'\.\.')

def from_file(filename):
    with open(filename) as file:
        str = file.readlines()
    return str

def get_lexicals(my_input):
    tmp = re.split(WHITESPACE, ''.join(my_input))
    final = []
    for item in tmp:
        last = 0
        matches = re.finditer(TOKEN_REGEX, item)
        hasItem = False
        for match in matches:
            hasItem = True
            if last != match.start():
                final.append(item[last:match.start()])
            final.append(match.group())
            last = match.end()
        if not hasItem:
            final.append(item)
    return final

def convert_to_tokens(my_input):
    tokens = []
    bad_tokens = []
    for lexical in my_input:
        matched = False
        if lexical in list(TOKENS):
            tokens.append(TOKENS[lexical])
            continue
        for expression in REGEX_TOKENS:
            if re.match(REGEX_TOKENS[expression], lexical):
                matched = True
                tokens.append(expression)
                break
        if not matched:
            bad_tokens.append(lexical)
    return tokens, bad_tokens

def analyse_input(input):
    # print('input:\n',input)
    lexicals = get_lexicals(input)
    # print('lexicals:\n',lexicals)
    tokens, invalid = convert_to_tokens(lexicals)
    # print('tokens:\n',tokens)
    # print('not tokens:\n',invalid)
    # if invalid:
    #     invalid = ','.join([f"'{x}'" for x in invalid])
    #     msg = f"Invalid input found. The tokens\n{invalid}\nwere not recognized."
    #     return msg
    single_list = list(zip(tokens, lexicals))
    return ' '.join([f'({x}, {y})' for x, y in single_list]) + '\n' + (f'Invalid Tokens: {", ".join(invalid)}' if invalid else 'SUCCESS')

print(analyse_input(from_file('Lexical_Input.txt')))