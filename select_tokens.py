def select_best_tokens(tokens, n,block_size):
    blocks = [tokens[i:i+block_size] for i in range(0, len(tokens), block_size)]
    print("blocks:", blocks)
    block_sums = [sum(block) for block in blocks]
    block_mean = sum(block_sums) / len(blocks)
    print("block_sums:", block_sums)
    print("block_mean:", block_mean)

tokens = [1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030]
top_n = 3
block_size = 3

best_tokens = select_best_tokens(tokens, top_n, block_size)
print(best_tokens)

import numpy as np

def select_important_tokens(data, block_size, top_n):
    """
    Bu fonksiyon, verilen bir diziyi bloklara ayırarak en önemli token'ları seçer.

    :param data: İşlem yapılacak dizi
    :param block_size: Blok boyutu
    :param top_n: Seçilecek en önemli blok sayısı
    :return: Seçilen en önemli token'ları içeren bir liste
    """
    blocks = [data[i:i + block_size] for i in range(0, len(data), block_size)]
    importance_scores = [sum(block) for block in blocks]  # Örnek olarak blok toplamını önem puanı olarak kullanıyoruz

    # En önemli blokları seç
    top_blocks_indices = np.argsort(importance_scores)[-top_n:]
    selected_tokens = [blocks[i] for i in top_blocks_indices]

    return selected_tokens

# Örnek veri seti
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
block_size = 2
top_n = 2

# En önemli token'ları seç
selected_tokens = select_important_tokens(data, block_size, top_n)
print("Seçilen En Önemli Token'lar:", selected_tokens)
