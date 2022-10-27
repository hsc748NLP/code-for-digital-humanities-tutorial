""" Modules for translation """
from onmt.translate.greedy_search import GreedySearch

__all__ = ['Translator', 'Translation', 'BeamSearch',
           'GNMTGlobalScorer', 'TranslationBuilder',
           'PenaltyBuilder', "DecodeStrategy", "GreedySearch"]
