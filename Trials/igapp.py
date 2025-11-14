import instaloader
import time


class InstagramScraper:
    """Utility class that wraps common Instaloader actions."""

    def __init__(self):
        """Initialize the Instaloader client."""
        self.L = instaloader.Instaloader(
            download_videos=True,
            download_video_thumbnails=True,
            download_geotags=True,
            download_comments=True,
            save_metadata=True,
            compress_json=False,
            max_connection_attempts=1,  # Fail fast
            request_timeout=10.0,  # 10 second timeout
        )

    def get_profile_info(self, username, calculate_engagement=False):
        """Retrieve profile information.

        Args:
            username: Instagram username.
            calculate_engagement: Whether to calculate engagement rate (may hit rate limits).
        """
        try:
            profile = instaloader.Profile.from_username(self.L.context, username)

            print("\n" + "=" * 50)
            print(f"📱 PROFILE INFORMATION: @{profile.username}")
            print("=" * 50)
            print(f"👤 Full Name: {profile.full_name}")
            print(f"👥 Followers: {profile.followers:,}")
            print(f"➕ Following: {profile.followees:,}")
            print(f"📸 Posts: {profile.mediacount}")
            print(f"📝 Bio: {profile.biography}")
            print(f"🔗 External URL: {profile.external_url}")
            print(f"🔒 Private Account: {'Yes' if profile.is_private else 'No'}")
            print(f"✅ Verified Account: {'Yes' if profile.is_verified else 'No'}")
            print(f"💼 Business Account: {'Yes' if profile.is_business_account else 'No'}")

            # Optionally calculate engagement (recent posts only).
            # NOTE: This can trigger Instagram rate limits.
            if calculate_engagement:
                try:
                    print("📊 Calculating engagement rate...")
                    total_engagement = 0
                    post_count = 0
                    for post in profile.get_posts():
                        if post_count >= 12:  # Last 12 posts
                            break
                        total_engagement += post.likes + post.comments
                        post_count += 1
                        time.sleep(0.5)  # Wait to reduce rate limit risk

                    if post_count > 0 and profile.followers > 0:
                        avg_engagement = total_engagement / post_count
                        engagement_rate = (avg_engagement / profile.followers) * 100
                        print(
                            f"📊 Engagement Rate: {engagement_rate:.2f}% (last {post_count} posts)"
                        )
                    else:
                        print("⚠️  Could not calculate engagement (insufficient posts)")
                except instaloader.exceptions.ConnectionException:
                    print(
                        "⚠️  Engagement calculation failed: Instagram rate limit (login recommended)"
                    )
                except instaloader.exceptions.LoginRequiredException:
                    print("⚠️  Engagement calculation failed: Login required")
                except Exception as exc:  # pragma: no cover - defensive logging
                    print(f"⚠️  Engagement calculation failed: {type(exc).__name__}")
            else:
                print("💡 Tip: Choose engagement calculation from the menu to enable it")

            print("=" * 50 + "\n")

            return profile
        except instaloader.exceptions.ProfileNotExistsException:
            print(f"❌ Error: user '{username}' was not found!")
            return None
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"❌ Error: {str(exc)}")
            return None

    def download_profile_posts(self, username, max_count=10):
        """Download posts from the given profile."""
        try:
            profile = instaloader.Profile.from_username(self.L.context, username)
            print(f"\n📥 Downloading posts for {username}...")

            count = 0
            for post in profile.get_posts():
                if count >= max_count:
                    break

                print(
                    f"  ⬇️  Post #{count + 1} - {post.date_local.strftime('%Y-%m-%d %H:%M')}"
                )
                print(f"      ❤️  Likes: {post.likes:,} | 💬 Comments: {post.comments}")

                # Download the post
                self.L.download_post(post, target=username)
                count += 1

            print(f"\n✅ {count} posts downloaded successfully!")

        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"❌ Error: {str(exc)}")

    def get_post_details(self, shortcode):
        """Retrieve details for a specific post."""
        try:
            post = instaloader.Post.from_shortcode(self.L.context, shortcode)

            print("\n" + "=" * 50)
            print("📸 POST DETAILS")
            print("=" * 50)
            print(f"👤 Owner: @{post.owner_username}")
            print(f"📅 Date: {post.date_local.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"❤️  Likes: {post.likes:,}")
            print(f"💬 Comments: {post.comments}")
            print(f"📝 Caption: {post.caption[:100]}..." if post.caption else "No caption")
            print(f"🔗 URL: https://www.instagram.com/p/{post.shortcode}/")
            print(f"📷 Photo: {'Yes' if not post.is_video else 'No'}")
            print(f"🎥 Video: {'Yes' if post.is_video else 'No'}")
            print(f"📱 Sidecar (Carousel): {'Yes' if post.typename == 'GraphSidecar' else 'No'}")

            if post.location:
                print(f"📍 Location: {post.location.name}")

            print("=" * 50 + "\n")

            return post
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"❌ Error: {str(exc)}")
            return None

    def get_hashtag_posts(self, hashtag, max_count=10):
        """List posts related to a specific hashtag."""
        try:
            print(f"\n🔍 Searching for posts with #{hashtag}...\n")

            hashtag_obj = instaloader.Hashtag.from_name(self.L.context, hashtag)
            print(f"📊 Total posts: {hashtag_obj.mediacount:,}\n")

            count = 0
            for post in hashtag_obj.get_posts():
                if count >= max_count:
                    break

                print(f"  {count + 1}. @{post.owner_username}")
                print(f"     ❤️  {post.likes:,} | 💬 {post.comments}")
                print(f"     📅 {post.date_local.strftime('%Y-%m-%d')}")
                print(f"     🔗 https://www.instagram.com/p/{post.shortcode}/\n")

                count += 1

            print(f"✅ Listed {count} posts!")

        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"❌ Error: {str(exc)}")

    def get_followers(self, username, max_count=20):
        """List followers for a user."""
        try:
            profile = instaloader.Profile.from_username(self.L.context, username)

            print(f"\n👥 Followers of {username}:\n")

            count = 0
            for follower in profile.get_followers():
                if count >= max_count:
                    break

                print(f"  {count + 1}. @{follower.username} - {follower.full_name}")
                count += 1

            print(f"\n✅ Listed {count} followers (Total: {profile.followers:,})")

        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"❌ Error: {str(exc)}")

    def get_followees(self, username, max_count=20):
        """List accounts the user follows."""
        try:
            profile = instaloader.Profile.from_username(self.L.context, username)

            print(f"\n➕ Accounts followed by {username}:\n")

            count = 0
            for followee in profile.get_followees():
                if count >= max_count:
                    break

                print(f"  {count + 1}. @{followee.username} - {followee.full_name}")
                count += 1

            print(f"\n✅ Listed {count} people (Total: {profile.followees:,})")

        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"❌ Error: {str(exc)}")

    def get_post_comments(self, shortcode, max_count=10):
        """Retrieve comments for a post."""
        try:
            post = instaloader.Post.from_shortcode(self.L.context, shortcode)

            print(f"\n💬 Post comments (Total: {post.comments}):\n")

            count = 0
            for comment in post.get_comments():
                if count >= max_count:
                    break

                print(f"  {count + 1}. @{comment.owner.username}:")
                print(f"     {comment.text[:100]}...")
                print(
                    f"     ❤️  {comment.likes_count} | 📅 {comment.created_at_utc.strftime('%Y-%m-%d %H:%M')}\n"
                )

                count += 1

            print(f"✅ Listed {count} comments!")

        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"❌ Error: {str(exc)}")

    def login(self, username, password):
        """Login to Instagram (required for some actions)."""
        try:
            self.L.login(username, password)
            print("✅ Logged in successfully!")
            return True
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"❌ Login error: {str(exc)}")
            return False

    def save_session(self, username):
        """Persist the authenticated session."""
        try:
            self.L.save_session_to_file(filename=f"session_{username}")
            print(f"✅ Session saved: session_{username}")
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"❌ Session save error: {str(exc)}")

    def load_session(self, username):
        """Load a previously saved session."""
        try:
            self.L.load_session_from_file(username, filename=f"session_{username}")
            print("✅ Session loaded!")
            return True
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"❌ Session load error: {str(exc)}")
            return False


def main():
    """Run the interactive menu."""
    scraper = InstagramScraper()

    while True:
        print("\n" + "=" * 50)
        print("📱 INSTAGRAM SCRAPER")
        print("=" * 50)
        print("1. View profile information")
        print("2. Download profile posts")
        print("3. View post details (with shortcode)")
        print("4. List hashtag posts")
        print("5. List followers")
        print("6. List following")
        print("7. View post comments")
        print("8. Log in (optional)")
        print("9. Profile info + engagement (slow, may hit rate limits)")
        print("0. Exit")
        print("=" * 50)

        choice = input("\nYour choice (0-9): ").strip()

        if choice == "1":
            username = input("Username: ").strip()
            scraper.get_profile_info(username, calculate_engagement=False)

        elif choice == "2":
            username = input("Username: ").strip()
            max_count = input("How many posts should be downloaded? (default 10): ").strip()
            max_count = int(max_count) if max_count else 10
            scraper.download_profile_posts(username, max_count)

        elif choice == "3":
            shortcode = input("Post shortcode (the /p/SHORTCODE/ part of the URL): ").strip()
            scraper.get_post_details(shortcode)

        elif choice == "4":
            hashtag = input("Hashtag (without #): ").strip()
            max_count = input("How many posts should be listed? (default 10): ").strip()
            max_count = int(max_count) if max_count else 10
            scraper.get_hashtag_posts(hashtag, max_count)

        elif choice == "5":
            username = input("Username: ").strip()
            max_count = input("How many followers should be listed? (default 20): ").strip()
            max_count = int(max_count) if max_count else 20
            scraper.get_followers(username, max_count)

        elif choice == "6":
            username = input("Username: ").strip()
            max_count = input("How many accounts should be listed? (default 20): ").strip()
            max_count = int(max_count) if max_count else 20
            scraper.get_followees(username, max_count)

        elif choice == "7":
            shortcode = input("Post shortcode: ").strip()
            max_count = input("How many comments should be displayed? (default 10): ").strip()
            max_count = int(max_count) if max_count else 10
            scraper.get_post_comments(shortcode, max_count)

        elif choice == "8":
            username = input("Your Instagram username: ").strip()
            password = input("Your Instagram password: ").strip()
            if scraper.login(username, password):
                scraper.save_session(username)

        elif choice == "9":
            username = input("Username: ").strip()
            print("\n⚠️  Warning: This action may hit Instagram rate limits!")
            print("💡 Logging in usually reduces errors.\n")
            scraper.get_profile_info(username, calculate_engagement=True)

        elif choice == "0":
            print("\n👋 See you later!")
            break

        else:
            print("❌ Invalid selection!")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    # Quick test (engagement calculation disabled)
    scraper = InstagramScraper()
    scraper.get_profile_info("emre.developer", calculate_engagement=False)

    # Uncomment to start the interactive menu:
    # main()
