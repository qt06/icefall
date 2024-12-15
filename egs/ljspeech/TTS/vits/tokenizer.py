# Copyright      2023-2024  Xiaomi Corp.        (authors: Zengwei Yao)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Dict, List
import jieba
from pypinyin import lazy_pinyin, Style
import tacotron_cleaner.cleaners

try:
    from piper_phonemize import phonemize_espeak
except Exception as ex:
    raise RuntimeError(
        f"{ex}\nPlease run\n"
        "pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html"
    )

from utils import intersperse


class Tokenizer(object):
    def __init__(self, tokens: str):
        """
        Args:
            tokens: the file that maps tokens to ids
        """
        # Parse token file
        self.token2id: Dict[str, int] = {}
        with open(tokens, "r", encoding="utf-8") as f:
            for line in f.readlines():
                info = line.rstrip().split()
                if len(info) == 1:
                    # case of space
                    token = " "
                    id = int(info[0])
                else:
                    token, id = info[0], int(info[1])
                assert token not in self.token2id, token
                self.token2id[token] = id

        # Refer to https://github.com/rhasspy/piper/blob/master/TRAINING.md
        self.pad_id = self.token2id["_"]  # padding
        self.sos_id = self.token2id["^"]  # beginning of an utterance (bos)
        self.eos_id = self.token2id["$"]  # end of an utterance (eos)
        self.space_id = self.token2id[" "]  # word separator (whitespace)

        self.vocab_size = len(self.token2id)

    # copy from https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/model/utils.py
    def convert_char_to_pinyin(self, text_list, polyphone=True):
        final_text_list = []
        custom_trans = str.maketrans(
            {"（": "(", "）": ")", "：": ":", "；": ",", ";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"}
        )  # add custom trans here, to address oov
    
        def is_chinese(c):
            return (
                "\u3100" <= c <= "\u9fff"  # common chinese characters
            )
    
        for text in text_list:
            char_list = []
            text = text.translate(custom_trans)
            for seg in jieba.cut(text):
                seg_byte_len = len(bytes(seg, "UTF-8"))
                if seg_byte_len == len(seg):  # if pure alphabets and symbols
                    if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                        char_list.append(" ")
                    char_list.extend(seg)
                elif polyphone and seg_byte_len == 3 * len(seg):  # if pure east asian characters
                    seg_ = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                    for i, c in enumerate(seg):
                        if is_chinese(c):
                            char_list.append(" ")
                        char_list.append(seg_[i])
                else:  # if mixed characters, alphabets and symbols
                    for c in seg:
                        if ord(c) < 256:
                            char_list.extend(c)
                        elif is_chinese(c):
                            char_list.append(" ")
                            char_list.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                        else:
                            char_list.append(c)
            final_text_list.append(char_list)
    
        return final_text_list

    def texts_to_token_ids(
        self,
        texts: List[str],
        intersperse_blank: bool = True,
        add_sos: bool = False,
        add_eos: bool = False,
        lang: str = "cmn",
    ) -> List[List[int]]:
        """
        Args:
          texts:
            A list of transcripts.
          intersperse_blank:
            Whether to intersperse blanks in the token sequence.
          add_sos:
            Whether to add sos token at the start.
          add_eos:
            Whether to add eos token at the end.
          lang:
            Language argument passed to phonemize_espeak().

        Returns:
          Return a list of token id list [utterance][token_id]
        """
        if lang in ['cmn', 'zh-cn']:
            tokens_list = self.convert_char_to_pinyin(texts)
            return self.tokens_to_token_ids(tokens_list, intersperse_blank, add_sos, add_eos)

        '''token_ids_list = []

        for text in texts:
            # Text normalization
            text = tacotron_cleaner.cleaners.custom_english_cleaners(text)
            # Convert to phonemes
            tokens_list = phonemize_espeak(text, lang)
            tokens = []
            for t in tokens_list:
                tokens.extend(t)

            token_ids = []
            for t in tokens:
                if t not in self.token2id:
                    logging.warning(f"Skip OOV {t}")
                    continue
                token_ids.append(self.token2id[t])

            if intersperse_blank:
                token_ids = intersperse(token_ids, self.pad_id)
            if add_sos:
                token_ids = [self.sos_id] + token_ids
            if add_eos:
                token_ids = token_ids + [self.eos_id]

            token_ids_list.append(token_ids)

        return token_ids_list'''

    def tokens_to_token_ids(
        self,
        tokens_list: List[str],
        intersperse_blank: bool = True,
        add_sos: bool = False,
        add_eos: bool = False,
    ) -> List[List[int]]:
        """
        Args:
          tokens_list:
            A list of token list, each corresponding to one utterance.
          intersperse_blank:
            Whether to intersperse blanks in the token sequence.
          add_sos:
            Whether to add sos token at the start.
          add_eos:
            Whether to add eos token at the end.

        Returns:
          Return a list of token id list [utterance][token_id]
        """
        token_ids_list = []

        for tokens in tokens_list:
            token_ids = []
            for t in tokens:
                if t not in self.token2id:
                    logging.warning(f"Skip OOV {t}")
                    continue
                token_ids.append(self.token2id[t])

            if intersperse_blank:
                token_ids = intersperse(token_ids, self.pad_id)
            if add_sos:
                token_ids = [self.sos_id] + token_ids
            if add_eos:
                token_ids = token_ids + [self.eos_id]

            token_ids_list.append(token_ids)

        return token_ids_list
