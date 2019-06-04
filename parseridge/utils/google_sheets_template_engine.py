from itertools import chain
from typing import List, NamedTuple

import pygsheets
from pygsheets import Cell

from parseridge.utils.logger import LoggerMixin


class GoogleSheetsTemplateEngine(LoggerMixin):
    class TemplateCell(NamedTuple):
        template_cell: Cell
        position: List[int]
        direction: str

        def inc_column(self, num=1):
            self.position[1] += num

        def inc_row(self, num=1):
            self.position[0] += num

        def reset_column(self):
            self.position[1] = 1

        def reset_row(self):
            self.position[0] = 1

    def __init__(self, worksheet_title, sheets_id, auth_file_path):
        self.sheets_id = sheets_id
        self.auth_file_path = auth_file_path
        self.title = worksheet_title

        self._cell_buffer = []

        self.spreadsheet = self._init_spreadsheet(self.sheets_id, self.auth_file_path)
        self.worksheet = self._create_worksheet(self.title)

        self.template_cells = self._parse_template()

    def update_variables(self, **kwargs):
        for var_name, var_value in kwargs.items():
            if var_name not in self.template_cells:
                continue

            template_cell = self.template_cells[var_name]
            if template_cell.direction == "fixed":
                template_cell.template_cell.value = str(var_value)
                self._cell_buffer.append(template_cell.template_cell)
            else:
                cell = Cell(tuple(template_cell.position))
                cell.value = str(var_value)

                # Update style
                if None not in template_cell.template_cell.color:
                    cell.color = template_cell.template_cell.color

                # Increase the position for the next entry of this cell
                if template_cell.direction == "row":
                    self.template_cells[var_name].inc_column()
                elif template_cell.direction == "column":
                    self.template_cells[var_name].inc_row()

                self._cell_buffer.append(cell)

    def sync(self):
        self.worksheet.update_cells(self._cell_buffer)
        self._cell_buffer = []

    def _create_worksheet(self, name):
        return self.spreadsheet.add_worksheet(
            title=name,
            src_worksheet=self.spreadsheet.worksheet_by_title("Template")
        )

    def _parse_template(self):
        template_cells = {}

        cells = self.worksheet.get_all_values(
            returnas="cell",
            include_tailing_empty=False,
            include_tailing_empty_rows=False
        )

        for cell in chain(*cells):
            if cell.value.startswith("<"):
                var_name = cell.value.replace("<", "").replace(">", "").lower()
                if "|" in var_name:
                    var_name, direction = var_name.split("|")
                else:
                    direction = "fixed"

                assert direction in ["row", "column", "fixed"]

                cell.unlink()
                cell.value = ""

                template_cells[var_name] = self.TemplateCell(
                    template_cell=cell,
                    position=[cell.row, cell.col],
                    direction=direction
                )

                self._cell_buffer.append(cell)

        self.sync()
        return template_cells

    @staticmethod
    def _init_spreadsheet(sheets_id, auth_file_path):
        return pygsheets.authorize(service_file=auth_file_path).open_by_key(sheets_id)
