import instaloader
from datetime import datetime
import os
import time

class InstagramScraper:
    def __init__(self):
        """Instaloader nesnesini başlatır"""
        self.L = instaloader.Instaloader(
            download_videos=True,
            download_video_thumbnails=True,
            download_geotags=True,
            download_comments=True,
            save_metadata=True,
            compress_json=False,
            max_connection_attempts=1,  # Hızlı fail için
            request_timeout=10.0  # 10 saniye timeout
        )
    
    def get_profile_info(self, username, calculate_engagement=False):
        """Profil bilgilerini getirir
        
        Args:
            username: Instagram kullanıcı adı
            calculate_engagement: Etkileşim oranını hesapla (rate limit riski var)
        """
        try:
            profile = instaloader.Profile.from_username(self.L.context, username)
            
            print("\n" + "="*50)
            print(f"📱 PROFIL BİLGİLERİ: @{profile.username}")
            print("="*50)
            print(f"👤 Tam İsim: {profile.full_name}")
            print(f"👥 Takipçi: {profile.followers:,}")
            print(f"➕ Takip Edilen: {profile.followees:,}")
            print(f"📸 Gönderi Sayısı: {profile.mediacount}")
            print(f"📝 Bio: {profile.biography}")
            print(f"🔗 Harici URL: {profile.external_url}")
            print(f"🔒 Gizli Hesap: {'Evet' if profile.is_private else 'Hayır'}")
            print(f"✅ Onaylı Hesap: {'Evet' if profile.is_verified else 'Hayır'}")
            print(f"💼 İş Hesabı: {'Evet' if profile.is_business_account else 'Hayır'}")
            
            # Etkileşim oranını manuel hesapla (son birkaç gönderi için)
            # NOT: Bu işlem Instagram'ın rate limit'ine takılabilir
            # Giriş yapmadan kullanımda sorun yaşanabilir
            if calculate_engagement:
                try:
                    print(f"📊 Etkileşim oranı hesaplanıyor...")
                    total_engagement = 0
                    post_count = 0
                    for post in profile.get_posts():
                        if post_count >= 12:  # Son 12 gönderi
                            break
                        total_engagement += post.likes + post.comments
                        post_count += 1
                        time.sleep(0.5)  # Rate limit için bekleme
                    
                    if post_count > 0 and profile.followers > 0:
                        avg_engagement = total_engagement / post_count
                        engagement_rate = (avg_engagement / profile.followers) * 100
                        print(f"📊 Etkileşim Oranı: {engagement_rate:.2f}% (son {post_count} gönderi)")
                    else:
                        print(f"⚠️  Etkileşim oranı hesaplanamadı (yeterli gönderi bulunamadı)")
                except instaloader.exceptions.ConnectionException as e:
                    print(f"⚠️  Etkileşim oranı hesaplanamadı: Instagram rate limit (Giriş yapmanız önerilir)")
                except instaloader.exceptions.LoginRequiredException:
                    print(f"⚠️  Etkileşim oranı hesaplanamadı: Giriş gerekli")
                except Exception as e:
                    print(f"⚠️  Etkileşim oranı hesaplanamadı: {type(e).__name__}")
            else:
                print(f"💡 İpucu: Etkileşim oranını hesaplamak için menüden seçim yapın")
            
            print("="*50 + "\n")
            
            return profile
        except instaloader.exceptions.ProfileNotExistsException:
            print(f"❌ Hata: '{username}' kullanıcısı bulunamadı!")
            return None
        except Exception as e:
            print(f"❌ Hata: {str(e)}")
            return None
    
    def download_profile_posts(self, username, max_count=10):
        """Kullanıcının gönderilerini indirir"""
        try:
            profile = instaloader.Profile.from_username(self.L.context, username)
            print(f"\n📥 {username} kullanıcısının gönderileri indiriliyor...")
            
            count = 0
            for post in profile.get_posts():
                if count >= max_count:
                    break
                
                print(f"  ⬇️  Post #{count+1} - {post.date_local.strftime('%Y-%m-%d %H:%M')}")
                print(f"      ❤️  Beğeni: {post.likes:,} | 💬 Yorum: {post.comments}")
                
                # Gönderiyi indir
                self.L.download_post(post, target=username)
                count += 1
            
            print(f"\n✅ {count} gönderi başarıyla indirildi!")
            
        except Exception as e:
            print(f"❌ Hata: {str(e)}")
    
    def get_post_details(self, shortcode):
        """Belirli bir gönderinin detaylarını getirir"""
        try:
            post = instaloader.Post.from_shortcode(self.L.context, shortcode)
            
            print("\n" + "="*50)
            print(f"📸 GÖNDERİ DETAYLARI")
            print("="*50)
            print(f"👤 Sahip: @{post.owner_username}")
            print(f"📅 Tarih: {post.date_local.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"❤️  Beğeni: {post.likes:,}")
            print(f"💬 Yorum: {post.comments}")
            print(f"📝 Açıklama: {post.caption[:100]}..." if post.caption else "Açıklama yok")
            print(f"🔗 URL: https://www.instagram.com/p/{post.shortcode}/")
            print(f"📷 Fotoğraf: {'Evet' if post.is_video == False else 'Hayır'}")
            print(f"🎥 Video: {'Evet' if post.is_video else 'Hayır'}")
            print(f"📱 Sidecar (Çoklu): {'Evet' if post.typename == 'GraphSidecar' else 'Hayır'}")
            
            if post.location:
                print(f"📍 Konum: {post.location.name}")
            
            print("="*50 + "\n")
            
            return post
        except Exception as e:
            print(f"❌ Hata: {str(e)}")
            return None
    
    def get_hashtag_posts(self, hashtag, max_count=10):
        """Belirli bir hashtag'e ait gönderileri getirir"""
        try:
            print(f"\n🔍 #{hashtag} hashtag'i için gönderiler aranıyor...\n")
            
            hashtag_obj = instaloader.Hashtag.from_name(self.L.context, hashtag)
            print(f"📊 Toplam gönderi sayısı: {hashtag_obj.mediacount:,}\n")
            
            count = 0
            for post in hashtag_obj.get_posts():
                if count >= max_count:
                    break
                
                print(f"  {count+1}. @{post.owner_username}")
                print(f"     ❤️  {post.likes:,} | 💬 {post.comments}")
                print(f"     📅 {post.date_local.strftime('%Y-%m-%d')}")
                print(f"     🔗 https://www.instagram.com/p/{post.shortcode}/\n")
                
                count += 1
            
            print(f"✅ {count} gönderi listelendi!")
            
        except Exception as e:
            print(f"❌ Hata: {str(e)}")
    
    def get_followers(self, username, max_count=20):
        """Kullanıcının takipçilerini listeler"""
        try:
            profile = instaloader.Profile.from_username(self.L.context, username)
            
            print(f"\n👥 {username} kullanıcısının takipçileri:\n")
            
            count = 0
            for follower in profile.get_followers():
                if count >= max_count:
                    break
                
                print(f"  {count+1}. @{follower.username} - {follower.full_name}")
                count += 1
            
            print(f"\n✅ {count} takipçi listelendi (Toplam: {profile.followers:,})")
            
        except Exception as e:
            print(f"❌ Hata: {str(e)}")
    
    def get_followees(self, username, max_count=20):
        """Kullanıcının takip ettiklerini listeler"""
        try:
            profile = instaloader.Profile.from_username(self.L.context, username)
            
            print(f"\n➕ {username} kullanıcısının takip ettikleri:\n")
            
            count = 0
            for followee in profile.get_followees():
                if count >= max_count:
                    break
                
                print(f"  {count+1}. @{followee.username} - {followee.full_name}")
                count += 1
            
            print(f"\n✅ {count} kişi listelendi (Toplam: {profile.followees:,})")
            
        except Exception as e:
            print(f"❌ Hata: {str(e)}")
    
    def get_post_comments(self, shortcode, max_count=10):
        """Bir gönderinin yorumlarını getirir"""
        try:
            post = instaloader.Post.from_shortcode(self.L.context, shortcode)
            
            print(f"\n💬 Gönderi yorumları (Toplam: {post.comments}):\n")
            
            count = 0
            for comment in post.get_comments():
                if count >= max_count:
                    break
                
                print(f"  {count+1}. @{comment.owner.username}:")
                print(f"     {comment.text[:100]}...")
                print(f"     ❤️  {comment.likes_count} | 📅 {comment.created_at_utc.strftime('%Y-%m-%d %H:%M')}\n")
                
                count += 1
            
            print(f"✅ {count} yorum listelendi!")
            
        except Exception as e:
            print(f"❌ Hata: {str(e)}")
    
    def login(self, username, password):
        """Instagram'a giriş yapar (bazı özellikler için gerekli)"""
        try:
            self.L.login(username, password)
            print("✅ Başarıyla giriş yapıldı!")
            return True
        except Exception as e:
            print(f"❌ Giriş hatası: {str(e)}")
            return False
    
    def save_session(self, username):
        """Oturum bilgilerini kaydeder"""
        try:
            self.L.save_session_to_file(filename=f"session_{username}")
            print(f"✅ Oturum kaydedildi: session_{username}")
        except Exception as e:
            print(f"❌ Oturum kaydetme hatası: {str(e)}")
    
    def load_session(self, username):
        """Kaydedilmiş oturumu yükler"""
        try:
            self.L.load_session_from_file(username, filename=f"session_{username}")
            print(f"✅ Oturum yüklendi!")
            return True
        except Exception as e:
            print(f"❌ Oturum yükleme hatası: {str(e)}")
            return False


def main():
    """Ana menü fonksiyonu"""
    scraper = InstagramScraper()
    
    while True:
        print("\n" + "="*50)
        print("📱 INSTAGRAM SCRAPER")
        print("="*50)
        print("1. Profil bilgilerini görüntüle")
        print("2. Profil gönderilerini indir")
        print("3. Gönderi detaylarını görüntüle (shortcode ile)")
        print("4. Hashtag gönderilerini listele")
        print("5. Takipçileri listele")
        print("6. Takip edilenleri listele")
        print("7. Gönderi yorumlarını görüntüle")
        print("8. Giriş yap (opsiyonel)")
        print("9. Profil bilgileri + Etkileşim oranı (yavaş, rate limit riski)")
        print("0. Çıkış")
        print("="*50)
        
        choice = input("\nSeçiminiz (0-9): ").strip()
        
        if choice == "1":
            username = input("Kullanıcı adı: ").strip()
            scraper.get_profile_info(username, calculate_engagement=False)
        
        elif choice == "2":
            username = input("Kullanıcı adı: ").strip()
            max_count = input("Kaç gönderi indirilsin? (varsayılan 10): ").strip()
            max_count = int(max_count) if max_count else 10
            scraper.download_profile_posts(username, max_count)
        
        elif choice == "3":
            shortcode = input("Gönderi shortcode'u (URL'deki /p/SHORTCODE/ kısmı): ").strip()
            scraper.get_post_details(shortcode)
        
        elif choice == "4":
            hashtag = input("Hashtag (# olmadan): ").strip()
            max_count = input("Kaç gönderi listelensin? (varsayılan 10): ").strip()
            max_count = int(max_count) if max_count else 10
            scraper.get_hashtag_posts(hashtag, max_count)
        
        elif choice == "5":
            username = input("Kullanıcı adı: ").strip()
            max_count = input("Kaç takipçi listelensin? (varsayılan 20): ").strip()
            max_count = int(max_count) if max_count else 20
            scraper.get_followers(username, max_count)
        
        elif choice == "6":
            username = input("Kullanıcı adı: ").strip()
            max_count = input("Kaç kişi listelensin? (varsayılan 20): ").strip()
            max_count = int(max_count) if max_count else 20
            scraper.get_followees(username, max_count)
        
        elif choice == "7":
            shortcode = input("Gönderi shortcode'u: ").strip()
            max_count = input("Kaç yorum görüntülensin? (varsayılan 10): ").strip()
            max_count = int(max_count) if max_count else 10
            scraper.get_post_comments(shortcode, max_count)
        
        elif choice == "8":
            username = input("Instagram kullanıcı adınız: ").strip()
            password = input("Instagram şifreniz: ").strip()
            if scraper.login(username, password):
                scraper.save_session(username)
        
        elif choice == "9":
            username = input("Kullanıcı adı: ").strip()
            print("\n⚠️  Uyarı: Bu işlem Instagram rate limit'ine takılabilir!")
            print("💡 Giriş yapmışsanız daha az sorun yaşarsınız.\n")
            scraper.get_profile_info(username, calculate_engagement=True)
        
        elif choice == "0":
            print("\n👋 Görüşürüz!")
            break
        
        else:
            print("❌ Geçersiz seçim!")
        
        input("\nDevam etmek için Enter'a basın...")


if __name__ == "__main__":
    # Hızlı test için (etkileşim oranı hesaplanmaz)
    scraper = InstagramScraper()
    scraper.get_profile_info("emre.developer", calculate_engagement=False)
    
    # İnteraktif menüyü başlatmak için yorumu kaldırın:
    # main()