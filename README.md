# Yapay Zeka Denemelerim

Bu repo, yapay zeka konusundaki öğrenimlerimi ve projelerimi paylaşmak amacıyla oluşturulmuştur.

## İçindekiler
- [Giriş](#giriş)
- [Projeler](#projeler)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Katkıda Bulunma](#katkıda-bulunma)
- [Lisans](#lisans)

## Giriş
Bu repo, Jupyter Notebook ve Python dillerinde yazılmış yapay zeka projelerini içermektedir. Amacım, bu alandaki bilgi ve deneyimlerimi paylaşmak ve diğer geliştiricilerle işbirliği yapmaktır.

## Projeler
Bu repo içerisindeki projelerle ilgili kısa bilgiler:

### 1. Genel Yapay Zeka Çalışmaları
Genel olarak öğrendiğim yapay zeka çalışmalarımı ve projeleri buradan paylaşıyor olacağım.

### 2. Weasel Projesi: Streamlit Entegrasyonu
<img src="https://user-images.githubusercontent.com/13643239/85388081-f2da8700-b545-11ea-9bd4-e303d3c5763c.png" width="300" height="auto" align="right" />

[Streamlit](https://streamlit.io) Python ile interaktif veri uygulamaları oluşturmak için bir framework'tür. [`spacy-streamlit`](https://github.com/explosion/spacy-streamlit) paketi, spaCy v3 ile Streamlit'i entegre etmenize yardımcı olur.

#### project.yml
[`project.yml`](project.yml) dosyası, proje tarafından gereken veri varlıklarını ve mevcut komutları ve iş akışlarını tanımlar. Detaylar için [Weasel dokümantasyonuna](https://github.com/explosion/weasel) bakabilirsiniz.

#### Komutlar
Belirlenen komutlar aşağıda listelenmiştir ve [`weasel run [name]`](https://github.com/explosion/weasel/tree/main/docs/cli.md#rocket-run) komutu ile çalıştırılabilir. Komutlar yalnızca girdileri değiştiğinde yeniden çalıştırılır.

| Komut | Açıklama |
| --- | --- |
| `download` | Modelleri indir |
| `visualize` | Streamlit kullanarak bir boru hattını etkileşimli olarak görselleştir |

#### Örnek
Detaylar için [`spacy-streamlit`](https://github.com/explosion/spacy-streamlit) projesine göz atabilirsiniz.

![Görselleştirici Ekran Görüntüsü](https://user-images.githubusercontent.com/13643239/85388081-f2da8700-b545-11ea-9bd4-e303d3c5763c.png)

## Kurulum
Projeleri çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

1. Bu repoyu klonlayın:
    ```sh
    git clone https://github.com/emredeveloper/Yapay-Zeka-Denemelerim.git
    ```
2. Gerekli bağımlılıkları yükleyin:
    ```sh
    pip install -r requirements.txt
    ```

## Kullanım
Projeleri çalıştırmak için Jupyter Notebook kullanabilirsiniz. Aşağıdaki komutu kullanarak Jupyter Notebook'u başlatabilirsiniz:
```sh
jupyter notebook
