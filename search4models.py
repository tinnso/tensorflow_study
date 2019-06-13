import xlrd
import os
# import sys

def search(path, keyword):
    print("search in: " + path)

    data = xlrd.open_workbook(path)

    sheet_count = data.nsheets

    # table = data.sheet_by_index(sheet_count - 1)
    for index in range(sheet_count):
        table = data.sheet_by_index(index)

        nrows = table.nrows
        nclos = table.ncols

        # print("sheet[%s] %d rows and %d columns" % (table.name, nrows, nclos))

        for i in range(nrows):
            for j in range(nclos):
                cellvalue = table.cell(i, j).value
                if cellvalue == keyword:
                    print(table.row_values(i))


def main():
    folder = "C:\\u_works\\03_project\\model list2"

    for root, dirs, files in os.walk(folder):
        for file in files:
            search(os.path.join(root, file), 'HT-Z9F')


if __name__ == '__main__':
    main()