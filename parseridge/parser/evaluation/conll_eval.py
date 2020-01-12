import io
from typing import Dict, Union, List

from parseridge.utils.logger import LoggerMixin

ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = range(10)


class CoNLLEvaluationScript(LoggerMixin):
    """
    Based on v. 1.0 of the CoNLL 2017 UD Parsing evaluation script by
    the Institute of Formal and Applied Linguistics (UFAL),
    Faculty of Mathematics and Physics, Charles University, Czech Republic.
    """

    class UDRepresentation:
        def __init__(self):
            # Characters of all the tokens in the whole file.
            # Whitespace between tokens is not included.
            self.characters = []
            # List of UDSpan instances with start&end indices into `characters`.
            self.tokens = []
            # List of UDWord instances.
            self.words = []
            # List of UDSpan instances with start&end indices into `characters`.
            self.sentences = []

    class UDSpan:
        def __init__(self, start, end):
            self.start = start
            # Note that end marks the first position **after the end** of span,
            # so we can use characters[start:end] or range(start, end).
            self.end = end

    class UDWord:
        def __init__(self, span, columns, is_multiword):
            # Span of this word (or MWT, see below) within
            # ud_representation.characters.
            self.span = span

            # 10 columns of the CoNLL-U file: ID, FORM, LEMMA,...
            self.columns = columns

            # is_multiword==True means that this word is part of a
            # multi-word token.  In that case, self.span marks the span of the
            # whole multi-word token.
            self.is_multiword = is_multiword

            # Reference to the UDWord instance representing the HEAD
            # (or None if root).
            self.parent = None

            # Let's ignore language-specific deprel subtypes.
            self.columns[DEPREL] = columns[DEPREL].split(":")[0]

    class UDError(Exception):
        pass

    class Score:
        def __init__(self, gold_total, system_total, correct, aligned_total=None):
            self.precision = correct / system_total if system_total else 0.0
            self.recall = correct / gold_total if gold_total else 0.0

            self.f1 = (
                2 * correct / (system_total + gold_total)
                if system_total + gold_total
                else 0.0
            )

            self.aligned_accuracy = (
                correct / aligned_total if aligned_total else aligned_total
            )

        def serialize(self):
            return {"f1": self.f1, "aligned_accuracy": self.aligned_accuracy}

    class AlignmentWord:
        def __init__(self, gold_word, system_word):
            self.gold_word = gold_word
            self.system_word = system_word
            self.gold_parent = None
            self.system_parent_gold_aligned = None

    class Alignment:
        def __init__(self, gold_words, system_words):
            self.gold_words = gold_words
            self.system_words = system_words
            self.matched_words = []
            self.matched_words_map = {}

        def append_aligned_words(self, gold_word, system_word):
            self.matched_words.append(
                CoNLLEvaluationScript.AlignmentWord(gold_word, system_word)
            )
            self.matched_words_map[system_word] = gold_word

        def fill_parents(self):
            """
            We represent root parents in both gold and system data by '0'.
            For gold data, we represent non-root parent by corresponding
            gold word. For system data, we represent non-root parent by
            either gold word aligned to parent system nodes, or by None if
            no gold words is aligned to the parent.
            """
            for words in self.matched_words:
                words.gold_parent = (
                    words.gold_word.parent if words.gold_word.parent is not None else 0
                )

                words.system_parent_gold_aligned = (
                    self.matched_words_map.get(words.system_word.parent, None)
                    if words.system_word.parent is not None
                    else 0
                )

    def evaluate(self, gold_ud, system_ud):
        # Check that underlying character sequences do match
        if gold_ud.characters != system_ud.characters:
            index = 0
            while gold_ud.characters[index] == system_ud.characters[index]:
                index += 1

            raise CoNLLEvaluationScript.UDError(
                f"The concatenation of tokens in the gold file and in "
                f"th system file differ!\n"
                + f"First 20 differing characters in gold file: "
                f"'{''.join(gold_ud.characters[index:index + 20])}' "
                f"and system file: "
                f"'{''.join(system_ud.characters[index:index + 20])}'"
            )

        # Align words
        alignment = self.align_words(gold_ud.words, system_ud.words)

        # Compute the F1-scores
        result = {
            "Tokens": self.spans_score(gold_ud.tokens, system_ud.tokens),
            "Sentences": self.spans_score(gold_ud.sentences, system_ud.sentences),
            "Words": self.alignment_score(alignment, None),
            "UPOS": self.alignment_score(alignment, lambda w, parent: w.columns[UPOS]),
            "XPOS": self.alignment_score(alignment, lambda w, parent: w.columns[XPOS]),
            "Feats": self.alignment_score(alignment, lambda w, parent: w.columns[FEATS]),
            "AllTags": self.alignment_score(
                alignment,
                lambda w, parent: (w.columns[UPOS], w.columns[XPOS], w.columns[FEATS]),
            ),
            "Lemmas": self.alignment_score(alignment, lambda w, parent: w.columns[LEMMA]),
            "UAS": self.alignment_score(alignment, lambda w, parent: parent),
            "LAS": self.alignment_score(
                alignment, lambda w, parent: (parent, w.columns[DEPREL])
            ),
        }

        return result

    @staticmethod
    def load_conllu(stream):
        ud = CoNLLEvaluationScript.UDRepresentation()

        # Load the CoNLL-U file
        index, sentence_start = 0, None
        line_number = 0
        while True:
            line = stream.readline()
            line_number += 1
            if not line:
                break
            line = line.rstrip("\r\n")

            # Handle sentence start boundaries
            if sentence_start is None:
                # Skip comments
                if line.startswith("#"):
                    continue
                # Start a new sentence
                ud.sentences.append(CoNLLEvaluationScript.UDSpan(index, 0))
                sentence_start = len(ud.words)
            if not line:
                # Add parent UDWord links and check there are no cycles
                def process_word(word):
                    if word.parent == "remapping":
                        raise CoNLLEvaluationScript.UDError(
                            "There is a cycle in a sentence"
                        )
                    if word.parent is None:
                        head = int(word.columns[HEAD])
                        if head > len(ud.words) - sentence_start:
                            raise CoNLLEvaluationScript.UDError(
                                f"HEAD '{word.columns[HEAD]}' "
                                f"points outside of the sentence"
                            )

                        if head:
                            parent = ud.words[sentence_start + head - 1]
                            word.parent = "remapping"
                            process_word(parent)
                            word.parent = parent

                for word in ud.words[sentence_start:]:
                    process_word(word)

                # Check there is a single root node
                if (
                    len([word for word in ud.words[sentence_start:] if word.parent is None])
                    != 1
                ):
                    raise CoNLLEvaluationScript.UDError(
                        f"There are multiple roots in a sentence. " f"(Line {line_number})."
                    )

                # End the sentence
                ud.sentences[-1].end = index
                sentence_start = None
                continue

            # Read next token/word
            columns = line.split("\t")
            if len(columns) != 10:
                raise CoNLLEvaluationScript.UDError(
                    f"The CoNLL-U line does not contain "
                    f"10 tab-separated columns: '{line}'"
                )

            # Skip empty nodes
            if "." in columns[ID]:
                continue

            # Delete spaces from FORM  so gold.characters == system.characters
            # even if one of them tokenizes the space.
            columns[FORM] = columns[FORM].replace(" ", "")
            if not columns[FORM]:
                raise CoNLLEvaluationScript.UDError(
                    "There is an empty FORM in the CoNLL-U file"
                )

            # Save token
            ud.characters.extend(columns[FORM])
            ud.tokens.append(
                CoNLLEvaluationScript.UDSpan(index, index + len(columns[FORM]))
            )
            index += len(columns[FORM])

            # Handle multi-word tokens to save word(s)
            if "-" in columns[ID]:
                try:
                    start, end = map(int, columns[ID].split("-"))
                except Exception:
                    raise CoNLLEvaluationScript.UDError(
                        "Cannot parse multi-word token ID '{}'".format(columns[ID])
                    )

                for _ in range(start, end + 1):
                    word_line = stream.readline().rstrip("\r\n")
                    line_number += 1
                    word_columns = word_line.split("\t")
                    if len(word_columns) != 10:
                        raise CoNLLEvaluationScript.UDError(
                            f"The CoNLL-U line does not contain "
                            f"10 tab-separated columns: '{word_line}'"
                        )

                    ud.words.append(
                        CoNLLEvaluationScript.UDWord(
                            ud.tokens[-1], word_columns, is_multiword=True
                        )
                    )

            # Basic tokens/words
            else:
                try:
                    word_id = int(columns[ID])
                except Exception:
                    raise CoNLLEvaluationScript.UDError(
                        "Cannot parse word ID '{}'".format(columns[ID])
                    )
                if word_id != len(ud.words) - sentence_start + 1:
                    raise CoNLLEvaluationScript.UDError(
                        f"Incorrect word ID '{columns[ID]}' "
                        f"for word '{columns[FORM]}', "
                        f"expected '{len(ud.words) - sentence_start + 1}'"
                    )

                try:
                    head_id = int(columns[HEAD])
                except Exception:
                    raise CoNLLEvaluationScript.UDError(
                        "Cannot parse HEAD '{}'".format(columns[HEAD])
                    )
                if head_id < 0:
                    raise CoNLLEvaluationScript.UDError("HEAD cannot be negative")

                ud.words.append(
                    CoNLLEvaluationScript.UDWord(ud.tokens[-1], columns, is_multiword=False)
                )

        if sentence_start is not None:
            raise CoNLLEvaluationScript.UDError(
                "The CoNLL-U file does not end with empty line"
            )

        return ud

    @staticmethod
    def spans_score(gold_spans, system_spans):
        correct, gi, si = 0, 0, 0
        while gi < len(gold_spans) and si < len(system_spans):
            if system_spans[si].start < gold_spans[gi].start:
                si += 1
            elif gold_spans[gi].start < system_spans[si].start:
                gi += 1
            else:
                correct += gold_spans[gi].end == system_spans[si].end
                si += 1
                gi += 1

        return CoNLLEvaluationScript.Score(len(gold_spans), len(system_spans), correct)

    @staticmethod
    def alignment_score(alignment, key_fn, weight_fn=lambda w: 1):
        gold, system, aligned, correct = 0, 0, 0, 0

        for word in alignment.gold_words:
            gold += weight_fn(word)

        for word in alignment.system_words:
            system += weight_fn(word)

        for words in alignment.matched_words:
            aligned += weight_fn(words.gold_word)

        if key_fn is None:
            # Return score for whole aligned words
            return CoNLLEvaluationScript.Score(gold, system, aligned)

        for words in alignment.matched_words:
            if key_fn(words.gold_word, words.gold_parent) == key_fn(
                words.system_word, words.system_parent_gold_aligned
            ):
                correct += weight_fn(words.gold_word)

        return CoNLLEvaluationScript.Score(gold, system, correct, aligned)

    @staticmethod
    def beyond_end(words, i, multiword_span_end):
        if i >= len(words):
            return True
        if words[i].is_multiword:
            return words[i].span.start >= multiword_span_end
        return words[i].span.end > multiword_span_end

    @staticmethod
    def extend_end(word, multiword_span_end):
        if word.is_multiword and word.span.end > multiword_span_end:
            return word.span.end
        return multiword_span_end

    def find_multiword_span(self, gold_words, system_words, gi, si):
        """
        We know gold_words[gi].is_multiword or system_words[si].is_multiword.
        Find the start of the multiword span (gs, ss), so the multiword span
        is minimal.
        """

        # Initialize multiword_span_end characters index.
        if gold_words[gi].is_multiword:
            multiword_span_end = gold_words[gi].span.end
            if (
                not system_words[si].is_multiword
                and system_words[si].span.start < gold_words[gi].span.start
            ):
                si += 1
        else:  # if system_words[si].is_multiword
            multiword_span_end = system_words[si].span.end
            if (
                not gold_words[gi].is_multiword
                and gold_words[gi].span.start < system_words[si].span.start
            ):
                gi += 1
        gs, ss = gi, si

        # Find the end of the multiword span (so both gi and si are pointing
        # to the word following the multiword span end).
        while not self.beyond_end(
            gold_words, gi, multiword_span_end
        ) or not self.beyond_end(system_words, si, multiword_span_end):
            if gi < len(gold_words) and (
                si >= len(system_words)
                or gold_words[gi].span.start <= system_words[si].span.start
            ):
                multiword_span_end = self.extend_end(gold_words[gi], multiword_span_end)
                gi += 1
            else:
                multiword_span_end = self.extend_end(system_words[si], multiword_span_end)
                si += 1
        return gs, ss, gi, si

    @staticmethod
    def compute_lcs(gold_words, system_words, gi, si, gs, ss):
        lcs = [[0] * (si - ss) for i in range(gi - gs)]
        for g in reversed(range(gi - gs)):
            for s in reversed(range(si - ss)):
                if (
                    gold_words[gs + g].columns[FORM].lower()
                    == system_words[ss + s].columns[FORM].lower()
                ):
                    lcs[g][s] = 1 + (
                        lcs[g + 1][s + 1] if g + 1 < gi - gs and s + 1 < si - ss else 0
                    )
                lcs[g][s] = max(lcs[g][s], lcs[g + 1][s] if g + 1 < gi - gs else 0)
                lcs[g][s] = max(lcs[g][s], lcs[g][s + 1] if s + 1 < si - ss else 0)
        return lcs

    def align_words(self, gold_words, system_words):
        alignment = CoNLLEvaluationScript.Alignment(gold_words, system_words)

        gi, si = 0, 0
        while gi < len(gold_words) and si < len(system_words):
            if gold_words[gi].is_multiword or system_words[si].is_multiword:
                gs, ss, gi, si = self.find_multiword_span(gold_words, system_words, gi, si)

                if si > ss and gi > gs:
                    lcs = self.compute_lcs(gold_words, system_words, gi, si, gs, ss)

                    # Store aligned words
                    s, g = 0, 0
                    while g < gi - gs and s < si - ss:
                        if (
                            gold_words[gs + g].columns[FORM].lower()
                            == system_words[ss + s].columns[FORM].lower()
                        ):
                            alignment.append_aligned_words(
                                gold_words[gs + g], system_words[ss + s]
                            )
                            g += 1
                            s += 1
                        elif lcs[g][s] == (lcs[g + 1][s] if g + 1 < gi - gs else 0):
                            g += 1
                        else:
                            s += 1
            else:
                # B: No multi-word token => align according to spans.
                if (gold_words[gi].span.start, gold_words[gi].span.end) == (
                    system_words[si].span.start,
                    system_words[si].span.end,
                ):
                    alignment.append_aligned_words(gold_words[gi], system_words[si])
                    gi += 1
                    si += 1
                elif gold_words[gi].span.start <= system_words[si].span.start:
                    gi += 1
                else:
                    si += 1

        alignment.fill_parents()

        return alignment

    def get_las_score_for_sentences(
        self, gold_sentences: List[str], predicted_sentences: List[str]
    ) -> Dict[str, Union[float, Dict[str, Dict[str, float]]]]:
        """
        Takes a list of gold an predicted sentence objects and computes the
        F1 LAS score between them.
        """

        gold_buffer = io.StringIO("".join(gold_sentences))
        pred_buffer = io.StringIO("".join(predicted_sentences))

        gold_connl = self.load_conllu(gold_buffer)
        pred_connl = self.load_conllu(pred_buffer)

        raw_scores = self.evaluate(gold_connl, pred_connl)

        scores = {
            "las": raw_scores["LAS"].f1 * 100,
            "uas": raw_scores["UAS"].f1 * 100,
            "all": {k: v.serialize() for k, v in raw_scores.items()},
        }

        return scores
