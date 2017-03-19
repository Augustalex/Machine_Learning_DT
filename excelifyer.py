from collections import defaultdict

import pandas


class Excelifyer:
    def __init__(self):
        self.table = defaultdict(lambda: defaultdict(lambda: -1))
        self.nextIndexAvailable = 0
        self.headers = defaultdict(lambda: '[' + self._get_next_header_index() + ']')

    def _get_next_header_index(self):
        index = self.nextIndexAvailable
        self.nextIndexAvailable += 1
        return str(index)

    def set_header(self, x, header):
        self.headers[x] = header

    def at_cell(self, x, y, value):
        self.table[x][y] = value

    def to_excel(self, filePath, sheet_name='data'):
        converted_columns = listify_and_default(self.table)
        columns = [listify_and_default(column) for column in converted_columns]
        headed_columns = dict((self.headers[index], columns[index]) for index in range(len(columns)))
        data_frame = pandas.DataFrame(headed_columns)
        writer = pandas.ExcelWriter(filePath, engine='xlsxwriter')
        data_frame.to_excel(writer, sheet_name=sheet_name)


def listify_and_default(dictionary, defaultValue='-'):
    res = []

    max_index = max(dictionary.keys())
    for index in range(0, max_index + 1):
        if index in dictionary:
            res.append(dictionary[index])
        else:
            res.append(defaultValue)

    return res


def test_case():
    doc = Excelifyer()

    doc.set_header(0, 'First column header')
    doc.at_cell(0, 0, 'A')
    doc.at_cell(0, 1, 'B')
    doc.at_cell(1, 0, 'C')
    doc.at_cell(1, 1, 'D')
    doc.to_excel('august_test_2.xlsx')

test_case()