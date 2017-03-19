from collections import defaultdict

import pandas


class Excelifyer:
    def __init__(self, use_column_headers=False, use_row_headers=False):
        self.use_column_headers = use_column_headers
        self.use_row_headers = use_row_headers
        self.table = defaultdict(lambda: defaultdict(lambda: -1))
        self.nextRowIndexAvailable = 0
        self.nextColumnIndexAvailable = 0
        # self.row_headers = defaultdict(lambda: '[' + self._get_next_row_header_index() + ']')
        # self.column_headers = defaultdict(lambda: '[' + self._get_next_column_header_index() + ']')
        self.row_headers = {}
        self.column_headers = {}
        self.width = 0
        self.height = 0

    def _get_next_row_header_index(self):
        index = self.nextRowIndexAvailable
        self.nextRowIndexAvailable += 1
        return str(index)

    def _get_next_column_header_index(self):
        index = self.nextRowIndexAvailable
        self.nextColumnIndexAvailable += 1
        return str(index)

    def set_column_header(self, x, header):
        self.column_headers[x] = header

    def set_row_header(self, y, header):
        self.row_headers[y] = header

    def at_cell(self, x, y, value):
        if self.use_row_headers:
            x += 1
        if self.use_column_headers:
            y += 1

        if x + 1 > self.width:
            self.width = x + 1
        if y + 1 > self.height:
            self.height = y + 1

        self.table[y][x] = value

    def at_row(self, y, header, values):
        self.at_cell(0, y, header)
        for i in range(0, len(values)):
            self.at_cell(i + 1, y, values[i])

    def at_column(self, x, header, values):
        self.set_column_header(x, header)
        for i in range(0, len(values)):
            self.at_cell(x, i, values[i])

    def to_excel(self, filePath, sheet_name='data'):
        self.prepare_headers()
        converted_columns = [column for column in self.listify_and_default(self.table, is_columns=True)]
        columns = [self.listify_and_default(column) for column in converted_columns]
        # headed_columns = dict((self.column_headers[index], columns[index]) for index in range(len(columns)))
        data_frame = pandas.DataFrame(columns)
        writer = pandas.ExcelWriter(filePath, engine='xlsxwriter')
        data_frame.to_excel(writer, sheet_name=sheet_name, header=False, index=False)

    def listify_and_default(self, dictionary, is_columns=False, defaultValue='-'):
        res = []

        limit = self.width if is_columns else self.height
        for index in range(0, limit):
            if index in dictionary:
                res.append(dictionary[index])
            elif not is_columns:
                res.append(defaultValue)

        return res

    def prepare_headers(self):
        if self.use_column_headers:
            for column_position in range(1, self.width):
                if (column_position - 1) in self.column_headers:
                    self.table[0][column_position] = self.column_headers[column_position - 1]
                else:
                    self.table[column_position][0] = '[' + str(self._get_next_column_header_index()) + ']'

        if self.use_row_headers:
            for row_position in range(1, self.height):
                if (row_position - 1) in self.row_headers:
                    self.table[row_position][0] = self.row_headers[row_position - 1]
                else:
                    self.table[row_position][0] = '[' + str(self._get_next_row_header_index()) + ']'



def test_case():
    # doc.set_column_header(0, 'First column header')
    doc = Excelifyer()
    doc.at_cell(1, 0, 'C')
    # doc.at_cell(0, 1, 'B')
    doc.at_cell(0, 0, 'A')
    doc.at_cell(1, 1, 'D')
    # doc.at_row(0, 'First row', ['A', 'C'])
    # doc.at_row(1, 'Second row', ['B', 'D'])
    doc.to_excel('august_test_2.xlsx')
