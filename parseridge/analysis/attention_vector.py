import time

from parseridge.utils.logger import LoggerMixin
from parseridge.utils.report import GoogleSheetsTemplateEngine


class AttentionVectorLogger(LoggerMixin):

    def __init__(self, vocabulary):
        sheets_id = "1pcY1Nu3eGNy_kBw0LHzvngfzxd78zK3zBLbRVN21Tvo"
        auth_file_path = "google_sheets_auth.json"
        name = str(int(time.time()))

        self.template_engine = GoogleSheetsTemplateEngine(name, sheets_id, auth_file_path)
        self.vocabulary = vocabulary

    def log_energies(self, sentences_batch, indices_batch, indices_lengths,
                     energies_batch, prefix=""):

        for sentence, indices, length, energies in zip(sentences_batch, indices_batch,
                                                       indices_lengths, energies_batch):
            if length == 0:
                continue

            token_ids = [
                sentence[idx] for idx in indices[:length]
            ]

            tokens = [self.vocabulary.get_item(id_) for id_ in token_ids]

            energies = energies[:length].squeeze().cpu().tolist()

            if not isinstance(energies, list):
                energies = [energies]

            out = []
            for token, energy in zip(tokens, energies):
                out.append(f"{token}[{energy:4f}]")

                self.template_engine.update_variables(
                    buffer_tokens=token,
                    buffer_energies=energy
                )

            self.template_engine.template_cells["buffer_tokens"].inc_row(2)
            self.template_engine.template_cells["buffer_energies"].inc_row(2)
            self.template_engine.template_cells["buffer_tokens"].reset_column()
            self.template_engine.template_cells["buffer_energies"].reset_column()

            self.template_engine.worksheet.add_rows(5)

        self.template_engine.sync()
