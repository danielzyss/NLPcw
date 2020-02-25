from parameters import *


class BPE:

    def __init__(self, data1, data2, n_symbol):
        self.data1 = data1
        self.data2 = data2
        self.EndOfWordChar = '·'
        self.n_char = n_symbol

    def LearnBPE(self):

        vocab1 = self.GetVocabulary(self.data1)
        vocab2 = self.GetVocabulary(self.data2)
        self.vocab = vocab1 + vocab2

        pairs, indices = self.PairStatistics(self.vocab)

        self.codes= []

        for i in tqdm.tqdm(range(self.n_char), desc="Learning BPE encoding"):
            most_frequent = max(pairs, key=lambda x: (pairs[x], x))
            changes = self.replace_pair(most_frequent, self.vocab, indices)
            self.update_pair_statistics(most_frequent, changes, pairs, indices)
            pairs[most_frequent] = 0

            self.codes.append('{0} {1}\n'.format(*most_frequent))


    def ApplyBPE(self, merges=-1, separator='@@'):

        self.bpe_codes = [tuple(item.strip('\r\n ').split(' ')) for (n, item) in enumerate(self.codes) if
                          (n < merges or merges == -1)]
        self.bpe_codes = dict([(code, i) for (i, code) in reversed(list(enumerate(self.bpe_codes)))])
        self.bpe_codes_reverse = dict([(pair[0] + pair[1], pair) for pair, i in self.bpe_codes.items()])
        self.separator = separator
        self.glossaries = []
        self.glossaries_regex = re.compile('^({})$'.format('|'.join(self.glossaries))) if self.glossaries else None
        self.cache = {}


    def GetVocabulary(self, data):
        vocab = Counter()
        for line in data:
            for word in line.split():
                if len(word)>1:
                    wordrepresentation = tuple(word[:-1]) + (word[-1]+self.EndOfWordChar,)
                else:
                    wordrepresentation = (word, self.EndOfWordChar)

                vocab[wordrepresentation]+=1
        vocab = dict(vocab)
        vocab = sorted(vocab.items(), key=lambda item:item[1], reverse=True)
        return vocab

    def PairStatistics(self, vocab):

        pairs = defaultdict(int)

        indices = defaultdict(lambda: defaultdict(int))

        for i, (word, freq) in enumerate(vocab):
            prev_char = word[0]
            for char in word[1:]:
                pairs[prev_char, char] += freq
                indices[prev_char, char][i] += 1
                prev_char = char

        return pairs, indices

    def replace_pair(self, pair, vocab, indices):

        """Replace all occurrences of a symbol pair ('A', 'B') with a new symbol 'AB'"""

        first, second = pair
        pair_str = ''.join(pair)
        pair_str = pair_str.replace('\\','\\\\')
        changes = []
        pattern = re.compile(r'(?<!\S)' + re.escape(first + ' ' + second) + r'(?!\S)')


        iterator = indices[pair].items()
        for j, freq in iterator:
            if freq < 1:
                continue
            word, freq = vocab[j]
            new_word = ' '.join(word)
            new_word = pattern.sub(pair_str, new_word)
            new_word = tuple(new_word.split(' '))

            vocab[j] = (new_word, freq)
            changes.append((j, new_word, word, freq))

        return changes

    def update_pair_statistics(self, pair, changed, stats, indices):
        """Minimally update the indices and frequency of symbol pairs
        if we merge a pair of symbols, only pairs that overlap with occurrences
        of this pair are affected, and need to be updated.
        """
        stats[pair] = 0
        indices[pair] = defaultdict(int)
        first, second = pair
        new_pair = first + second
        for j, word, old_word, freq in changed:

            # find all instances of pair, and update frequency/indices around it
            i = 0
            while True:
                # find first symbol
                try:
                    i = old_word.index(first, i)
                except ValueError:
                    break
                # if first symbol is followed by second symbol, we've found an occurrence of pair (old_word[i:i+2])
                if i < len(old_word) - 1 and old_word[i + 1] == second:
                    # assuming a symbol sequence "A B C", if "B C" is merged, reduce the frequency of "A B"
                    if i:
                        prev = old_word[i - 1:i + 1]
                        stats[prev] -= freq
                        indices[prev][j] -= 1
                    if i < len(old_word) - 2:
                        # assuming a symbol sequence "A B C B", if "B C" is merged, reduce the frequency of "C B".
                        # however, skip this if the sequence is A B C B C, because the frequency of "C B" will be reduced by the previous code block
                        if old_word[i + 2] != first or i >= len(old_word) - 3 or old_word[i + 3] != second:
                            nex = old_word[i + 1:i + 3]
                            stats[nex] -= freq
                            indices[nex][j] -= 1
                    i += 2
                else:
                    i += 1

            i = 0
            while True:
                try:
                    # find new pair
                    i = word.index(new_pair, i)
                except ValueError:
                    break
                # assuming a symbol sequence "A BC D", if "B C" is merged, increase the frequency of "A BC"
                if i:
                    prev = word[i - 1:i + 1]
                    stats[prev] += freq
                    indices[prev][j] += 1
                # assuming a symbol sequence "A BC B", if "B C" is merged, increase the frequency of "BC B"
                # however, if the sequence is A BC BC, skip this step because the count of "BC BC" will be incremented by the previous code block
                if i < len(word) - 1 and word[i + 1] != new_pair:
                    nex = word[i:i + 2]
                    stats[nex] += freq
                    indices[nex][j] += 1
                i += 1

    def segment(self, sentence, dropout=0):
        segments = self.segment_tokens(sentence.strip('\r\n ').split(' '), dropout)
        return ' '.join(segments)

    def segment_tokens(self, tokens, dropout=0):
        """segment a sequence of tokens with BPE encoding"""
        output = []
        for word in tokens:
            # eliminate double spaces
            if not word:
                continue
            new_word = [out for segment in self._isolate_glossaries(word)
                        for out in self.encode(segment,
                                          self.bpe_codes,
                                          self.bpe_codes_reverse,
                                          self.vocab,
                                          self.separator,
                                          self.cache,
                                          self.glossaries_regex,
                                          dropout)]

            for item in new_word[:-1]:
                output.append(item + self.separator)
            output.append(new_word[-1])

        return output

    def encode(self, orig, bpe_codes, bpe_codes_reverse, vocab, separator, cache, glossaries_regex=None, dropout=0):
        """Encode word based on list of BPE merge operations, which are applied consecutively
        """

        if not dropout and orig in cache:
            return cache[orig]

        if glossaries_regex and glossaries_regex.match(orig):
            cache[orig] = (orig,)
            return (orig,)

        if len(orig) == 1:
            return orig

        word = list(orig[:-1]) + [orig[-1] + self.EndOfWordChar]


        while len(word) > 1:

            # get list of symbol pairs; optionally apply dropout
            pairs = [(bpe_codes[pair], i, pair) for (i, pair) in enumerate(zip(word, word[1:])) if
                     (not dropout or random.random() > dropout) and pair in bpe_codes]

            if not pairs:
                break

            # get first merge operation in list of BPE codes
            bigram = min(pairs)[2]

            # find start position of all pairs that we want to merge
            positions = [i for (rank, i, pair) in pairs if pair == bigram]

            i = 0
            new_word = []
            bigram = ''.join(bigram)
            for j in positions:
                # merges are invalid if they start before current position. This can happen if there are overlapping pairs: (x x x -> xx x)
                if j < i:
                    continue
                new_word.extend(word[i:j])  # all symbols before merged pair
                new_word.append(bigram)  # merged pair
                i = j + 2  # continue after merged pair
            new_word.extend(word[i:])  # add all symbols until end of word
            word = new_word

        # don't print end-of-word symbols
        if word[-1] == '</w>':
            word = word[:-1]
        elif word[-1].endswith('</w>'):
            word[-1] = word[-1][:-4]

        word = tuple(word)
        if vocab:
            word = self.check_vocab_and_split(word, bpe_codes_reverse, vocab, separator)

        cache[orig] = word
        return word

    def recursive_split(self, segment, bpe_codes, vocab, separator, final=False):
        """Recursively split segment into smaller units (by reversing BPE merges)
        until all units are either in-vocabulary, or cannot be split futher."""

        try:
            if final:
                left, right = bpe_codes[segment + '</w>']
                right = right[:-4]
            else:
                left, right = bpe_codes[segment]
        except:
            # sys.stderr.write('cannot split {0} further.\n'.format(segment))
            yield segment
            return

        if left + separator in vocab:
            yield left
        else:
            for item in self.recursive_split(left, bpe_codes, vocab, separator, False):
                yield item

        if (final and right in vocab) or (not final and right + separator in vocab):
            yield right
        else:
            for item in self.recursive_split(right, bpe_codes, vocab, separator, final):
                yield item

    def check_vocab_and_split(self, orig, bpe_codes, vocab, separator):
        """Check for each segment in word if it is in-vocabulary,
        and segment OOV segments into smaller units by reversing the BPE merge operations"""

        out = []

        for segment in orig[:-1]:
            if segment + separator in vocab:
                out.append(segment)
            else:
                # sys.stderr.write('OOV: {0}\n'.format(segment))
                for item in self.recursive_split(segment, bpe_codes, vocab, separator, False):
                    out.append(item)

        segment = orig[-1]
        if segment in vocab:
            out.append(segment)
        else:
            # sys.stderr.write('OOV: {0}\n'.format(segment))
            for item in self.recursive_split(segment, bpe_codes, vocab, separator, True):
                out.append(item)

        return out

    def _isolate_glossaries(self, word):
        word_segments = [word]
        for gloss in self.glossaries:
            word_segments = [out_segments for segment in word_segments
                             for out_segments in self.isolate_glossary(segment, gloss)]
        return word_segments

    def isolate_glossary(self, word, glossary):
        """
        Isolate a glossary present inside a word.
        Returns a list of subwords. In which all 'glossary' glossaries are isolated
        For example, if 'USA' is the glossary and '1934USABUSA' the word, the return value is:
            ['1934', 'USA', 'B', 'USA']
        """
        # regex equivalent of (if word == glossary or glossary not in word)
        if re.match('^' + glossary + '$', word) or not re.search(glossary, word):
            return [word]
        else:
            segments = re.split(r'({})'.format(glossary), word)
            segments, ending = segments[:-1], segments[-1]
            segments = list(filter(None, segments))  # Remove empty strings in regex group.
            return segments + [ending.strip('\r\n ')] if ending != '' else segments

    def ApplyEncoding(self, data):
        new_data = []
        for d in tqdm.tqdm(data, desc="encoding data to BPE format"):
            new_data.append(self.segment(d))
            print(new_data)
        return new_data
