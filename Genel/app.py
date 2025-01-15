from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.textfield import MDTextField
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.label import MDLabel
from kivymd.uix.menu import MDDropdownMenu
from kivy.metrics import dp
from kivy.properties import StringProperty
from kivymd.uix.dialog import MDDialog
from kivymd.uix.snackbar import Snackbar

class DemographicsForm(MDScreen):
    gender = StringProperty("Cinsiyet Seçin")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = MDBoxLayout(orientation='vertical', spacing=dp(20), padding=dp(20))
        self.add_widget(self.layout)

        self.layout.add_widget(MDLabel(
            text="Demografik Bilgi Formu",
            font_style="H4",
            halign="center",
            size_hint_y=1,
            height=dp(100)
        ))

        self.fields = {
            'name': 'Ad Soyad',
            'age': 'Yaş',
            'email': 'E-posta',
            'phone': 'Telefon'
        }

        self.inputs = {}

        for key, label in self.fields.items():
            text_input = MDTextField(
                hint_text=label,
                mode="rectangle",
                size_hint_x=None,
                width=dp(300)
            )
            self.inputs[key] = text_input
            self.layout.add_widget(text_input)

        self.gender_button = MDRaisedButton(
            text=self.gender,
            size_hint=(None, None),
            width=dp(300),
            on_release=self.gender_menu_open
        )
        self.layout.add_widget(self.gender_button)

        gender_items = [
            {"viewclass": "OneLineListItem", "text": f"{i}", "on_release": lambda x=f"{i}": self.gender_menu_callback(x)}
            for i in ("Erkek", "Kadın", "Diğer")
        ]
        self.gender_menu = MDDropdownMenu(
            caller=self.gender_button,
            items=gender_items,
            width_mult=3,
        )

        button_layout = MDBoxLayout(spacing=dp(10), size_hint=(None, None), width=dp(300), height=dp(50))
        self.submit_btn = MDRaisedButton(
            text='Gönder',
            on_press=self.on_submit,
            md_bg_color=self.theme_cls.primary_color,
        )
        self.cancel_btn = MDRaisedButton(
            text='İptal',
            on_press=self.on_cancel,
            md_bg_color=self.theme_cls.error_color,
        )
        button_layout.add_widget(self.submit_btn)
        button_layout.add_widget(self.cancel_btn)
        self.layout.add_widget(button_layout)

    def gender_menu_open(self, instance):
        self.gender_menu.open()

    def gender_menu_callback(self, text_item):
        self.gender_button.text = text_item
        self.gender = text_item
        self.gender_menu.dismiss()

    def on_submit(self, instance):
        data = {key: input.text.strip() for key, input in self.inputs.items()}
        data['gender'] = self.gender
        
        if self.validate_data(data):
            print(data)
            self.show_confirmation_dialog(data)
        else:
            Snackbar(text="Lütfen tüm alanları doldurun.").open()

    def validate_data(self, data):
        return all(data.values()) and data['gender'] != "Cinsiyet Seçin"

    def show_confirmation_dialog(self, data):
        content = f"Ad Soyad: {data['name']}\nYaş: {data['age']}\nE-posta: {data['email']}\nTelefon: {data['phone']}\nCinsiyet: {data['gender']}"
        dialog = MDDialog(
            title="Bilgilerinizi onaylayın",
            text=content,
            buttons=[
                MDRaisedButton(
                    text="Onayla", on_release=lambda x: self.confirm_and_close(dialog)
                ),
                MDRaisedButton(
                    text="İptal", on_release=lambda x: dialog.dismiss()
                ),
            ],
        )
        dialog.open()

    def confirm_and_close(self, dialog):
        dialog.dismiss()
        Snackbar(text="Bilgileriniz başarıyla kaydedildi.").open()
        MDApp.get_running_app().stop()

    def on_cancel(self, instance):
        MDApp.get_running_app().stop()

class DemographicsApp(MDApp):
    def build(self):
        self.theme_cls.primary_palette = "Blue"
        self.theme_cls.theme_style = "Light"
        return DemographicsForm()

if __name__ == '__main__':
    DemographicsApp().run()