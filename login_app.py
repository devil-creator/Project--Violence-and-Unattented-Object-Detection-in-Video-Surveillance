import os
import json
import subprocess
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ObjectProperty
from kivy.lang import Builder

Builder.load_file('login_screen.kv')

CREDENTIALS_FILE = 'user_credentials.json'

class LoginScreen(Screen):
    def attempt_login(self):
        username = self.ids.username.text
        password = self.ids.password.text

        if not os.path.exists(CREDENTIALS_FILE):
            self.ids.status.text = "No account found. Please sign up."
            return

        with open(CREDENTIALS_FILE, 'r') as f:
            try:
                credentials = json.load(f)
            except:
                self.ids.status.text = "Error reading credentials"
                return

        if username in credentials and credentials[username] == password:
            self.ids.status.text = "Login successful!"
            self.launch_main_app()
        else:
            self.ids.status.text = "Invalid credentials"

    def launch_main_app(self):
        try:
            subprocess.Popen(['python', 'maintest.py'])
            App.get_running_app().stop()
        except Exception as e:
            self.ids.status.text = f"Error: {str(e)}"

    def show_signup(self):
        self.manager.current = 'signup'

class SignupScreen(Screen):
    def create_account(self):
        username = self.ids.new_username.text
        password = self.ids.new_password.text
        confirm = self.ids.confirm_password.text

        if not username or not password:
            self.ids.signup_status.text = "All fields required"
            return

        if password != confirm:
            self.ids.signup_status.text = "Passwords don't match"
            return

        credentials = {}
        if os.path.exists(CREDENTIALS_FILE):
            with open(CREDENTIALS_FILE, 'r') as f:
                credentials = json.load(f)

        if username in credentials:
            self.ids.signup_status.text = "Username exists"
            return

        credentials[username] = password
        with open(CREDENTIALS_FILE, 'w') as f:
            json.dump(credentials, f)

        self.ids.signup_status.text = "Account created!"
        self.manager.current = 'login'

class LoginApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(LoginScreen(name='login'))
        sm.add_widget(SignupScreen(name='signup'))
        return sm

if __name__ == '__main__':
    LoginApp().run()