import math
from collections import Counter
from typing import List, Set, Tuple
import numpy as np

class StringSimilarity:
    """Gelişmiş metin benzerliği ölçüm algoritmalarını içeren sınıf"""
    
    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """
        İki string arasındaki Levenshtein mesafesini hesaplar.
        
        Args:
            s1: İlk metin
            s2: İkinci metin
            
        Returns:
            int: İki metin arasındaki minimum düzenleme mesafesi
        """
        if len(s1) < len(s2):
            return StringSimilarity.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1] + [0] * len(s2)
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row[j + 1] = min(insertions, deletions, substitutions)
            prev_row = curr_row
        
        return prev_row[-1]
    
    @staticmethod
    def levenshtein_ratio(s1: str, s2: str) -> float:
        """
        Levenshtein benzerliğini 0-1 arasında normalize eder.
        
        Args:
            s1: İlk metin
            s2: İkinci metin
            
        Returns:
            float: 0-1 arasında benzerlik oranı
        """
        distance = StringSimilarity.levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        return 1.0 if max_len == 0 else 1 - (distance / max_len)
    
    @staticmethod
    def jaro_similarity(s1: str, s2: str) -> float:
        """
        Jaro benzerliğini hesaplar.
        
        Args:
            s1: İlk metin
            s2: İkinci metin
            
        Returns:
            float: 0-1 arasında Jaro benzerlik skoru
        """
        s1, s2 = s1.upper(), s2.upper()
        
        if s1 == s2:
            return 1.0
        
        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0
        
        match_distance = max(len1, len2) // 2 - 1
        if match_distance < 0:
            match_distance = 0
        
        s1_matches = [False] * len1
        s2_matches = [False] * len2
        matches = 0
        transpositions = 0
        
        for i in range(len1):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len2)
            
            for j in range(start, end):
                if s2_matches[j] or s1[i] != s2[j]:
                    continue
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break
        
        if matches == 0:
            return 0.0
        
        k = 0
        for i in range(len1):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1
        
        return (matches/len1 + matches/len2 + 
                (matches - transpositions/2)/matches) / 3
    
    @staticmethod
    def jaro_winkler_similarity(s1: str, s2: str, scaling: float = 0.1) -> float:
        """
        Jaro-Winkler benzerliğini hesaplar (prefix bonusu ile).
        
        Args:
            s1: İlk metin
            s2: İkinci metin
            scaling: Ön ek ağırlığı (genellikle 0.1)
            
        Returns:
            float: 0-1 arasında Jaro-Winkler benzerlik skoru
        """
        jaro = StringSimilarity.jaro_similarity(s1, s2)
        
        if jaro < 0.7:
            return jaro
        
        prefix = 0
        min_len = min(len(s1), len(s2))
        for i in range(min_len):
            if s1[i].upper() == s2[i].upper():
                prefix += 1
            else:
                break
        
        return jaro + (prefix * scaling * (1 - jaro))
    
    @staticmethod
    def jaccard_similarity(s1: str, s2: str, ngram_size: int = 1) -> float:
        """
        Jaccard benzerliğini hesaplar.
        
        Args:
            s1: İlk metin
            s2: İkinci metin
            ngram_size: N-gram boyutu (1=kelime, 2=bigram, vb.)
            
        Returns:
            float: 0-1 arasında Jaccard benzerlik skoru
        """
        def get_ngrams(text: str, n: int) -> Set[str]:
            words = text.split()
            if n == 1:
                return set(words)
            return {' '.join(words[i:i+n]) for i in range(len(words) - n + 1)}
        
        set1 = get_ngrams(s1, ngram_size)
        set2 = get_ngrams(s2, ngram_size)
        
        if not set1 and not set2:
            return 1.0
            
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def cosine_similarity(s1: str, s2: str) -> float:
        """
        Kosinüs benzerliğini hesaplar.
        
        Args:
            s1: İlk metin
            s2: İkinci metin
            
        Returns:
            float: 0-1 arasında kosinüs benzerlik skoru
        """
        def text_to_vector(text: str) -> Counter:
            words = text.lower().split()
            return Counter(words)
        
        vec1 = text_to_vector(s1)
        vec2 = text_to_vector(s2)
        
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])
        
        sum1 = sum([vec1[x]**2 for x in vec1.keys()])
        sum2 = sum([vec2[x]**2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)
        
        if not denominator:
            return 0.0
        return float(numerator) / denominator
    
    @staticmethod
    def compare_multiple(texts: List[str], method: str = 'levenshtein') -> List[List[float]]:
        """
        Birden fazla metin arasındaki benzerlikleri karşılaştırır.
        
        Args:
            texts: Karşılaştırılacak metinlerin listesi
            method: Kullanılacak benzerlik metodu ('levenshtein', 'jaro', 'jaro_winkler', 'jaccard', 'cosine')
            
        Returns:
            List[List[float]]: Benzerlik matrisi
        """
        n = len(texts)
        similarity_matrix = [[0.0] * n for _ in range(n)]
        
        method_func = {
            'levenshtein': StringSimilarity.levenshtein_ratio,
            'jaro': StringSimilarity.jaro_similarity,
            'jaro_winkler': StringSimilarity.jaro_winkler_similarity,
            'jaccard': StringSimilarity.jaccard_similarity,
            'cosine': StringSimilarity.cosine_similarity
        }.get(method, StringSimilarity.levenshtein_ratio)
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity = 1.0
                else:
                    similarity = method_func(texts[i], texts[j])
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity
        
        return similarity_matrix
