import pandas as pd


# Load the Excel file
def import_dataset(sheet_name):
    excel_file = pd.ExcelFile('(Optimization, AICS301)Lecture 07 Example of Backpropagation.xlsx')

    df = excel_file.parse(sheet_name, header = None)

    return df


def main():
    sheet_name = 'Data'
    df = import_dataset(sheet_name)

    df.to_csv(f'{sheet_name}.csv', index = False, encoding = "utf-8-sig")

