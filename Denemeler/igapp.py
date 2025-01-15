import instaloader

# İndirmek istediğiniz Instagram kullanıcısının adını girin
profile_name = "emre.developer"

# Instaloader sınıfından bir nesne oluşturun
L = instaloader.Instaloader()

# Profili yükleyin
profile = instaloader.Profile.from_username(L.context, profile_name)

# Profil bilgilerini yazdırın
print(f"Kullanıcı Adı: {profile.username}")
print(f"Takipçi sayısı: {profile.followers}")
print(f"Takip edilen kişi sayısı: {profile.followees}")
print(f"Bio: {profile.biography}")
print(f"Gönderi sayısı: {profile.mediacount}")

# Profildeki tüm gönderileri indirin (sadece medyayı indirir, likes vb. bilgileri içermez)
for post in profile.get_posts():
    L.download_post(post, target=profile_name)

print(f"{profile_name} kullanıcısının tüm gönderileri başarıyla indirildi.")